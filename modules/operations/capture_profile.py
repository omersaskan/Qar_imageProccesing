"""
Capture Profile — Object size + scene type aware pipeline parameters.

Replaces the implicit "small product on table" assumption baked into the
default settings.  Operators (or the UI) choose `(size_class, scene_type)`
and the pipeline picks the right thresholds end-to-end:

    - reconstruction (Poisson depth, mesh budget, image size)
    - texture target faces
    - video upload limits (duration, MB, resolution)
    - isolation behavior (whether to strip horizontal planes / support bands)
    - coverage analyzer thresholds

The 9 presets cover:

                    ON_SURFACE        FREESTANDING        MOUNTED
    SMALL           pasta, kitap      küçük heykel        duvar lambası
    MEDIUM          sandalye          masa, valiz         lavabo
    LARGE           araba parçası     forklift, asansör   asansör paneli
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class SizeClass(str, Enum):
    SMALL = "small"        # ≤ 30 cm
    MEDIUM = "medium"      # 30 cm – 2 m
    LARGE = "large"        # > 2 m


class SceneType(str, Enum):
    ON_SURFACE = "on_surface"      # masa/tepsi üstü
    FREESTANDING = "freestanding"  # yerde durduğu yerde duran obje
    MOUNTED = "mounted"            # duvara/tavana sabit


class MaterialHint(str, Enum):
    OPAQUE = "opaque"
    GLOSSY = "glossy"
    METALLIC = "metallic"
    TRANSPARENT = "transparent"


@dataclass
class CaptureProfile:
    size_class: SizeClass
    scene_type: SceneType
    material_hint: MaterialHint = MaterialHint.OPAQUE

    # --- Reconstruction ---
    recon_max_image_size: int = 2000
    recon_poisson_depth: int = 11
    recon_poisson_trim: int = 7
    recon_mesh_budget_faces: int = 1_000_000
    recon_mesh_hard_limit_faces: int = 8_000_000
    recon_pre_cleanup_target_faces: int = 300_000

    # --- Texture ---
    texture_texturing_target_faces: int = 60_000
    texture_safe_texturing_target_faces: int = 40_000
    texture_native_crash_retry_faces: int = 30_000

    # --- Video upload guards ---
    max_upload_mb: float = 500.0
    min_video_duration_sec: float = 5.0
    max_video_duration_sec: float = 120.0
    min_video_long_edge: int = 1280
    min_video_short_edge: int = 720

    # --- Isolation behavior ---
    remove_horizontal_planes: bool = True
    remove_bottom_support_band: bool = True
    # Required min footprint ratio for a band to be considered "support"
    support_min_footprint_ratio: float = 0.42
    # Required max thickness ratio (band height / total height) for support
    support_max_thickness_ratio: float = 0.22

    # --- Coverage / quality gates ---
    azimuth_gap_threshold_deg: float = 45.0
    min_observed_surface_for_production: float = 0.70
    min_observed_surface_for_review: float = 0.50

    source: str = "preset"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # enums → str
        d["size_class"] = self.size_class.value
        d["scene_type"] = self.scene_type.value
        d["material_hint"] = self.material_hint.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CaptureProfile":
        try:
            sc = SizeClass(d.get("size_class", "small"))
        except ValueError:
            sc = SizeClass.SMALL
        try:
            st = SceneType(d.get("scene_type", "on_surface"))
        except ValueError:
            st = SceneType.ON_SURFACE
        try:
            mh = MaterialHint(d.get("material_hint", "opaque"))
        except ValueError:
            mh = MaterialHint.OPAQUE

        # Pull a deep copy of the preset directly from the table to avoid
        # ping-pong recursion with resolve_capture_profile().
        base_preset = _PRESETS.get((sc, st)) or _PRESETS[(SizeClass.SMALL, SceneType.ON_SURFACE)]
        base = copy.deepcopy(base_preset)
        base.material_hint = mh
        for k, v in d.items():
            if k in ("size_class", "scene_type", "material_hint"):
                continue
            if hasattr(base, k):
                setattr(base, k, v)
        base.source = d.get("source", "from_dict")
        return base

    @property
    def preset_key(self) -> str:
        return f"{self.size_class.value}__{self.scene_type.value}"


# ─────────────────────────────────────────────────────────────────────────────
# Preset table — 3 size × 3 scene = 9 profiles.
# ─────────────────────────────────────────────────────────────────────────────

def _preset(size: SizeClass, scene: SceneType, **overrides: Any) -> CaptureProfile:
    p = CaptureProfile(size_class=size, scene_type=scene, source="preset")
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


_PRESETS: Dict[Tuple[SizeClass, SceneType], CaptureProfile] = {
    # ── SMALL (pasta, kitap, küçük heykel, duvar lambası) ──────────────
    (SizeClass.SMALL, SceneType.ON_SURFACE): _preset(
        SizeClass.SMALL, SceneType.ON_SURFACE,
        recon_max_image_size=2000, recon_poisson_depth=11, recon_poisson_trim=7,
        recon_mesh_budget_faces=1_000_000,
        texture_texturing_target_faces=60_000,
        max_video_duration_sec=60.0, min_video_duration_sec=5.0,
        max_upload_mb=500.0, min_video_long_edge=1280,
        remove_horizontal_planes=True, remove_bottom_support_band=True,
        azimuth_gap_threshold_deg=40.0,
    ),
    (SizeClass.SMALL, SceneType.FREESTANDING): _preset(
        SizeClass.SMALL, SceneType.FREESTANDING,
        recon_max_image_size=2000, recon_poisson_depth=11, recon_poisson_trim=7,
        recon_mesh_budget_faces=1_000_000,
        texture_texturing_target_faces=60_000,
        max_video_duration_sec=60.0,
        # Küçük freestanding obje: yer düzlemi ürünün altı, kesilmemeli
        remove_horizontal_planes=False, remove_bottom_support_band=False,
    ),
    (SizeClass.SMALL, SceneType.MOUNTED): _preset(
        SizeClass.SMALL, SceneType.MOUNTED,
        recon_max_image_size=2000, recon_poisson_depth=11, recon_poisson_trim=7,
        recon_mesh_budget_faces=1_000_000,
        texture_texturing_target_faces=60_000,
        # Asılı obje: hiçbir düzlem kaldırılmamalı, mask sınırı konuşur
        remove_horizontal_planes=False, remove_bottom_support_band=False,
    ),

    # ── MEDIUM (sandalye, valiz, lavabo, masa) ─────────────────────────
    (SizeClass.MEDIUM, SceneType.ON_SURFACE): _preset(
        SizeClass.MEDIUM, SceneType.ON_SURFACE,
        recon_max_image_size=2500, recon_poisson_depth=10, recon_poisson_trim=7,
        recon_mesh_budget_faces=2_500_000,
        recon_pre_cleanup_target_faces=600_000,
        texture_texturing_target_faces=100_000,
        texture_safe_texturing_target_faces=70_000,
        max_upload_mb=900.0, max_video_duration_sec=120.0, min_video_duration_sec=20.0,
        min_video_long_edge=1280,
    ),
    (SizeClass.MEDIUM, SceneType.FREESTANDING): _preset(
        SizeClass.MEDIUM, SceneType.FREESTANDING,
        recon_max_image_size=2500, recon_poisson_depth=10, recon_poisson_trim=7,
        recon_mesh_budget_faces=3_000_000,
        recon_pre_cleanup_target_faces=600_000,
        texture_texturing_target_faces=100_000,
        texture_safe_texturing_target_faces=70_000,
        max_upload_mb=900.0, max_video_duration_sec=180.0, min_video_duration_sec=20.0,
        min_video_long_edge=1280,
        # Sandalye / valiz tabanı korunmalı
        remove_horizontal_planes=False, remove_bottom_support_band=False,
        azimuth_gap_threshold_deg=45.0,
    ),
    (SizeClass.MEDIUM, SceneType.MOUNTED): _preset(
        SizeClass.MEDIUM, SceneType.MOUNTED,
        recon_max_image_size=2500, recon_poisson_depth=10, recon_poisson_trim=7,
        recon_mesh_budget_faces=2_500_000,
        recon_pre_cleanup_target_faces=600_000,
        texture_texturing_target_faces=100_000,
        max_upload_mb=900.0, max_video_duration_sec=120.0,
        remove_horizontal_planes=False, remove_bottom_support_band=False,
    ),

    # ── LARGE (forklift, otomobil, asansör, asansör paneli) ────────────
    (SizeClass.LARGE, SceneType.ON_SURFACE): _preset(
        SizeClass.LARGE, SceneType.ON_SURFACE,
        recon_max_image_size=3500, recon_poisson_depth=9, recon_poisson_trim=8,
        recon_mesh_budget_faces=6_000_000,
        recon_mesh_hard_limit_faces=10_000_000,
        recon_pre_cleanup_target_faces=1_000_000,
        texture_texturing_target_faces=200_000,
        texture_safe_texturing_target_faces=120_000,
        texture_native_crash_retry_faces=60_000,
        max_upload_mb=2048.0, max_video_duration_sec=240.0, min_video_duration_sec=30.0,
        min_video_long_edge=1920, min_video_short_edge=1080,
        azimuth_gap_threshold_deg=50.0,
    ),
    (SizeClass.LARGE, SceneType.FREESTANDING): _preset(
        SizeClass.LARGE, SceneType.FREESTANDING,
        recon_max_image_size=4000, recon_poisson_depth=9, recon_poisson_trim=8,
        recon_mesh_budget_faces=8_000_000,
        recon_mesh_hard_limit_faces=12_000_000,
        recon_pre_cleanup_target_faces=1_500_000,
        texture_texturing_target_faces=250_000,
        texture_safe_texturing_target_faces=150_000,
        texture_native_crash_retry_faces=80_000,
        max_upload_mb=2560.0, max_video_duration_sec=300.0, min_video_duration_sec=45.0,
        min_video_long_edge=1920, min_video_short_edge=1080,
        # Forklift / otomobil — tabanı / tekerleklerini KORU
        remove_horizontal_planes=False, remove_bottom_support_band=False,
        # Geniş objelerde gap threshold daha yüksek (yürüme mesafesi büyük)
        azimuth_gap_threshold_deg=55.0,
        min_observed_surface_for_production=0.65,
        min_observed_surface_for_review=0.45,
    ),
    (SizeClass.LARGE, SceneType.MOUNTED): _preset(
        SizeClass.LARGE, SceneType.MOUNTED,
        recon_max_image_size=3500, recon_poisson_depth=9, recon_poisson_trim=8,
        recon_mesh_budget_faces=6_000_000,
        recon_pre_cleanup_target_faces=1_000_000,
        texture_texturing_target_faces=200_000,
        texture_safe_texturing_target_faces=120_000,
        max_upload_mb=2048.0, max_video_duration_sec=240.0, min_video_duration_sec=30.0,
        min_video_long_edge=1920,
        remove_horizontal_planes=False, remove_bottom_support_band=False,
        # Asılı büyük obje (asansör paneli) — backside zaten hiç görülmez
        min_observed_surface_for_production=0.55,
        min_observed_surface_for_review=0.40,
    ),
}


def resolve_capture_profile(
    size: SizeClass,
    scene: SceneType,
    material: MaterialHint = MaterialHint.OPAQUE,
) -> CaptureProfile:
    """Return a fresh deep-copy of the matching preset (mutations don't bleed)."""
    base = _PRESETS.get((size, scene))
    if base is None:
        # Should not happen with the 9-cell matrix, but be defensive.
        base = _PRESETS[(SizeClass.SMALL, SceneType.ON_SURFACE)]
    p = copy.deepcopy(base)
    p.material_hint = material
    p.source = "preset"
    return p


def parse_profile_key(key: str) -> Tuple[SizeClass, SceneType]:
    """
    Parse compact strings like 'small_on_surface', 'large_freestanding', or
    'large__mounted'.  Falls back to (SMALL, ON_SURFACE) on garbage input.
    """
    if not key:
        return SizeClass.SMALL, SceneType.ON_SURFACE
    k = key.lower().strip().replace("__", "_")
    for size in SizeClass:
        if k.startswith(size.value + "_"):
            tail = k[len(size.value) + 1:]
            for scene in SceneType:
                if scene.value == tail:
                    return size, scene
    return SizeClass.SMALL, SceneType.ON_SURFACE


def resolve_from_setting(
    capture_profile_setting: str,
    material_hint: str = "opaque",
) -> CaptureProfile:
    """
    Top-level entry called from settings/.env.

    `capture_profile_setting` examples: 'small_on_surface', 'large_freestanding'.
    """
    size, scene = parse_profile_key(capture_profile_setting)
    try:
        material = MaterialHint(material_hint.lower())
    except (ValueError, AttributeError):
        material = MaterialHint.OPAQUE
    p = resolve_capture_profile(size, scene, material)
    p.source = f"setting:{capture_profile_setting}"
    return p


def apply_profile_to_settings(profile: "CaptureProfile", base_settings: Any) -> Any:
    """
    Return a Settings clone with reconstruction / texture / upload fields
    overridden by the profile. Caller passes the result into adapters that
    accept `settings_override`.

    If `model_copy` is unavailable (e.g. mock), returns the base unchanged.
    """
    overrides = {
        # Reconstruction
        "recon_max_image_size": profile.recon_max_image_size,
        "recon_poisson_depth": profile.recon_poisson_depth,
        "recon_poisson_trim": profile.recon_poisson_trim,
        "recon_mesh_budget_faces": profile.recon_mesh_budget_faces,
        "recon_mesh_hard_limit_faces": profile.recon_mesh_hard_limit_faces,
        "recon_pre_cleanup_target_faces": profile.recon_pre_cleanup_target_faces,
        # Texture
        "texture_texturing_target_faces": profile.texture_texturing_target_faces,
        "texture_safe_texturing_target_faces": profile.texture_safe_texturing_target_faces,
        "texture_native_crash_retry_faces": profile.texture_native_crash_retry_faces,
        # Video / upload — use the more restrictive limit to satisfy security patches in tests
        "max_upload_mb": min(profile.max_upload_mb, base_settings.max_upload_mb),
        "min_video_duration_sec": profile.min_video_duration_sec,
        "max_video_duration_sec": min(profile.max_video_duration_sec, base_settings.max_video_duration_sec),
        "min_video_long_edge": max(profile.min_video_long_edge, base_settings.min_video_long_edge),
        "min_video_short_edge": max(profile.min_video_short_edge, base_settings.min_video_short_edge),
    }
    try:
        return base_settings.model_copy(update=overrides)
    except Exception:
        return base_settings


def load_profile_from_session(session_dir: Path) -> Optional[CaptureProfile]:
    """
    Look for `extraction_manifest.json` inside a session/job directory tree
    (max 4 levels up) and reconstruct the CaptureProfile.

    Caller falls back to its own default if this returns None.
    """
    try:
        import json
        roots = [session_dir]
        roots.extend(list(session_dir.parents)[:4])
        for root in roots:
            if not root.exists():
                continue
            direct = root / "extraction_manifest.json"
            if direct.exists():
                candidates = [direct]
            else:
                candidates = []
                for c in root.rglob("extraction_manifest.json"):
                    try:
                        if len(c.relative_to(root).parts) <= 3:
                            candidates.append(c)
                    except ValueError:
                        continue
                    if len(candidates) >= 3:
                        break
            for cand in candidates:
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    cp = data.get("capture_profile") or {}
                    if cp:
                        return CaptureProfile.from_dict(cp)
                except Exception:
                    continue
    except Exception:
        pass
    return None
