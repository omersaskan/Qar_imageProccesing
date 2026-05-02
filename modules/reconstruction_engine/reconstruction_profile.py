"""
Reconstruction Profile — derive material / size / scene / motion classifications
from upstream signals (extraction_manifest, capture_gate, frame stats).

This is the *input* to reconstruction_preset_resolver: tell the resolver what
kind of capture it's looking at, the resolver picks safe COLMAP/OpenMVS knobs.

Deterministic by design — same inputs always yield the same profile.
Confidence score reflects how many sub-signals corroborated the classification
(useful for downstream gates; the resolver itself uses the labels not the score).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MaterialProfile(str, Enum):
    MATTE = "matte"
    GLOSSY = "glossy"
    TRANSPARENT_REFLECTIVE = "transparent_reflective"
    LOW_TEXTURE = "low_texture"
    UNKNOWN = "unknown"


class SizeProfile(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    UNKNOWN = "unknown"


class SceneProfile(str, Enum):
    CONTROLLED = "controlled"
    CLUTTERED = "cluttered"
    LOW_LIGHT = "low_light"
    UNKNOWN = "unknown"


class MotionProfile(str, Enum):
    STABLE_ORBIT = "stable_orbit"
    FAST_MOTION = "fast_motion"
    STATIC_POOR = "static_poor"
    UNEVEN = "uneven"
    UNKNOWN = "unknown"


@dataclass
class ReconstructionProfile:
    material_profile: MaterialProfile = MaterialProfile.UNKNOWN
    size_profile: SizeProfile = SizeProfile.UNKNOWN
    scene_profile: SceneProfile = SceneProfile.UNKNOWN
    motion_profile: MotionProfile = MotionProfile.UNKNOWN
    confidence: float = 0.0
    signals_used: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "material_profile": self.material_profile.value,
            "size_profile": self.size_profile.value,
            "scene_profile": self.scene_profile.value,
            "motion_profile": self.motion_profile.value,
            "confidence": round(self.confidence, 3),
            "signals_used": list(self.signals_used),
            "reasons": list(self.reasons),
        }


def _derive_material(
    capture_profile: Optional[Dict[str, Any]],
    color_profile: Optional[Dict[str, Any]],
    frame_count: int,
) -> tuple:
    """Returns (MaterialProfile, contributing_signal_count, reasons)."""
    reasons: List[str] = []
    signals = 0

    if capture_profile:
        signals += 1
        hint = (capture_profile.get("material_hint") or "").lower()
        if hint == "transparent":
            reasons.append("capture_profile.material_hint=transparent")
            return MaterialProfile.TRANSPARENT_REFLECTIVE, signals, reasons
        if hint == "glossy":
            reasons.append("capture_profile.material_hint=glossy")
            return MaterialProfile.GLOSSY, signals, reasons
        if hint == "metallic":
            reasons.append("capture_profile.material_hint=metallic → glossy class")
            return MaterialProfile.GLOSSY, signals, reasons

    # Color-driven low-texture hint: very dark monochrome OR very bright
    # monochrome surfaces yield poor SIFT features.
    if color_profile:
        signals += 1
        cat = (color_profile.get("category") or "").lower()
        if cat in ("black",) and frame_count > 0:
            reasons.append("color_profile.category=black → low_texture risk")
            return MaterialProfile.LOW_TEXTURE, signals, reasons
        if cat in ("white_cream",) and frame_count > 0:
            # White/cream is feature-light only when uniform; without a stronger
            # signal we keep it MATTE rather than over-classifying.
            reasons.append("color_profile.category=white_cream (kept matte)")
            return MaterialProfile.MATTE, signals, reasons

    if capture_profile and capture_profile.get("material_hint") == "opaque":
        reasons.append("opaque hint, no color signal → matte default")
        return MaterialProfile.MATTE, signals, reasons

    return MaterialProfile.UNKNOWN, signals, reasons


def _derive_size(capture_profile: Optional[Dict[str, Any]]) -> tuple:
    if not capture_profile:
        return SizeProfile.UNKNOWN, 0, []
    sz = (capture_profile.get("size_class") or "").lower()
    if sz == "small":
        return SizeProfile.SMALL, 1, ["capture_profile.size_class=small"]
    if sz == "medium":
        return SizeProfile.MEDIUM, 1, ["capture_profile.size_class=medium"]
    if sz == "large":
        return SizeProfile.LARGE, 1, ["capture_profile.size_class=large"]
    return SizeProfile.UNKNOWN, 0, []


def _derive_scene(
    capture_gate: Optional[Dict[str, Any]],
    color_profile: Optional[Dict[str, Any]],
) -> tuple:
    """
    LOW_LIGHT: dark mean RGB AND low blur median (proxy for under-exposure).
    CLUTTERED: gate detects high background dominance / chromatic leakage.
    CONTROLLED: gate decision pass + multi_height ≥ 0.5.
    """
    reasons: List[str] = []
    signals = 0

    if color_profile:
        signals += 1
        product_rgb = color_profile.get("product_rgb") or [0, 0, 0]
        mean_v = sum(product_rgb) / max(len(product_rgb), 1) if product_rgb else 0
        if mean_v < 50:
            reasons.append(f"product_rgb mean {mean_v:.0f} < 50 → low_light")
            return SceneProfile.LOW_LIGHT, signals, reasons

    if capture_gate:
        signals += 1
        decision = (capture_gate.get("decision") or "").lower()
        elev = capture_gate.get("elevation") or {}
        mh = float(elev.get("multi_height_score", 0.0) or 0.0)

        # blur median is a proxy for under-exposure / texture poverty
        blur = capture_gate.get("blur") or {}
        median_blur = float(blur.get("median_score", 0.0) or 0.0)
        if median_blur > 0 and median_blur < 30 and decision != "pass":
            reasons.append(f"median_blur {median_blur:.1f} < 30 → low_light")
            return SceneProfile.LOW_LIGHT, signals, reasons

        if decision == "pass" and mh >= 0.5:
            reasons.append("gate=pass + multi_height ≥ 0.5 → controlled")
            return SceneProfile.CONTROLLED, signals, reasons

    return SceneProfile.UNKNOWN, signals, reasons


def _derive_motion(
    capture_gate: Optional[Dict[str, Any]],
    adaptive_sampling: Optional[Dict[str, Any]],
) -> tuple:
    """
    STATIC_POOR: orbit_progress very low + many adaptive_sampling SKIP_STATIC.
    FAST_MOTION: blur burst ratio high OR many KEEP_MOTION_BURST.
    STABLE_ORBIT: orbit_progress ≥ 0.6 AND blur burst < 10%.
    UNEVEN: high static_run + sporadic motion.
    """
    reasons: List[str] = []
    signals = 0

    if capture_gate:
        signals += 1
        az = capture_gate.get("azimuth") or {}
        blur = capture_gate.get("blur") or {}
        orbit = float(az.get("cumulative_orbit_progress", 0.0) or 0.0)
        burst_ratio = float(blur.get("burst_ratio", 0.0) or 0.0)
        static_run = int(az.get("max_consecutive_static_frames", 0) or 0)
        n_frames = int(az.get("frame_count", 0) or 0)
        static_run_ratio = static_run / max(n_frames, 1)

        if orbit < 0.20 and static_run_ratio > 0.30:
            reasons.append(f"orbit {orbit:.2f} <0.20 + static_run {static_run_ratio:.0%} → static_poor")
            return MotionProfile.STATIC_POOR, signals, reasons
        if burst_ratio >= 0.20:
            reasons.append(f"burst_ratio {burst_ratio:.0%} ≥ 20% → fast_motion")
            return MotionProfile.FAST_MOTION, signals, reasons
        if static_run_ratio > 0.30 and orbit >= 0.20:
            reasons.append(f"high static_run {static_run_ratio:.0%} mid orbit → uneven")
            return MotionProfile.UNEVEN, signals, reasons
        if orbit >= 0.60 and burst_ratio < 0.10:
            reasons.append(f"orbit {orbit:.2f} ≥ 0.60 + burst {burst_ratio:.0%} <10% → stable_orbit")
            return MotionProfile.STABLE_ORBIT, signals, reasons

    if adaptive_sampling and adaptive_sampling.get("enabled"):
        signals += 1
        stats = (adaptive_sampling.get("stats") or {}).get("decisions", {}) or {}
        bursts = int(stats.get("keep_motion_burst", 0) or 0)
        statics = int(stats.get("skip_static", 0) or 0)
        kept = int((adaptive_sampling.get("stats") or {}).get("kept_count", 0) or 0)
        if bursts > 0 and bursts >= max(2, kept * 0.30):
            reasons.append(f"adaptive_sampling bursts {bursts}/{kept} → fast_motion")
            return MotionProfile.FAST_MOTION, signals, reasons
        if statics > 0 and statics >= kept * 2:
            reasons.append(f"adaptive_sampling statics {statics}/{kept} → static_poor")
            return MotionProfile.STATIC_POOR, signals, reasons

    return MotionProfile.UNKNOWN, signals, reasons


def derive_profile(
    extraction_manifest: Optional[Dict[str, Any]] = None,
    selected_keyframe_count: Optional[int] = None,
) -> ReconstructionProfile:
    """
    Deterministic profile inference from upstream signals.

    `extraction_manifest` is the dict written by frame_extractor (the same
    file scorecard / capture-gate endpoint reads).  `selected_keyframe_count`
    overrides the frame count inferred from the manifest.
    """
    em = extraction_manifest or {}
    capture_profile = em.get("capture_profile")
    color_profile = em.get("color_profile")
    capture_gate = em.get("capture_gate")
    adaptive_sampling = em.get("adaptive_sampling")
    frame_count = int(selected_keyframe_count or em.get("saved_count") or em.get("frame_count") or 0)

    rep = ReconstructionProfile()
    total_signals = 0

    mat, mat_sig, mat_reasons = _derive_material(capture_profile, color_profile, frame_count)
    rep.material_profile = mat
    total_signals += mat_sig
    rep.reasons.extend(mat_reasons)
    if mat_sig:
        rep.signals_used.append("material")

    sz, sz_sig, sz_reasons = _derive_size(capture_profile)
    rep.size_profile = sz
    total_signals += sz_sig
    rep.reasons.extend(sz_reasons)
    if sz_sig:
        rep.signals_used.append("size")

    sc, sc_sig, sc_reasons = _derive_scene(capture_gate, color_profile)
    rep.scene_profile = sc
    total_signals += sc_sig
    rep.reasons.extend(sc_reasons)
    if sc_sig:
        rep.signals_used.append("scene")

    mt, mt_sig, mt_reasons = _derive_motion(capture_gate, adaptive_sampling)
    rep.motion_profile = mt
    total_signals += mt_sig
    rep.reasons.extend(mt_reasons)
    if mt_sig:
        rep.signals_used.append("motion")

    # Confidence: each signal worth 0.25; cap at 1.0
    rep.confidence = float(min(1.0, total_signals * 0.25))

    return rep
