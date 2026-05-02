"""
Reconstruction Preset Resolver — profile → COLMAP/OpenMVS preset bundle.

Conservative-by-design.  v1 is intentionally narrow:

    Preset names:
        - baseline                — env defaults (always-available fallback)
        - profile_safe            — generic profile-aware safer tune
        - low_texture_safe        — lower SIFT thresholds, sequential matcher
        - low_light_safe          — smaller image size, longer matcher iters
        - texture_retry_safe      — reduced TextureMesh budget + threads

We don't ship 36-cell aggressive matrix yet (that's Sprint 4 v2).  v1 just
proves the wiring + preset table can grow.

Every preset is a `dict` shaped like:
    {
        "name": "<preset name>",
        "colmap": {
            "feature_quality": "low|medium|high",
            "matcher_type": "exhaustive|sequential",
            "max_image_size": int,
            "mapper_min_num_matches": int,
            "patchmatch_resolution_level": int,   # 0=full, 1=half, 2=quarter
        },
        "openmvs": {
            "texture_resolution": int,   # max texture atlas dim
            "max_threads": int,          # 0=auto
            "enable_texture_retry": bool,
        },
        "rationale": "...",
    }
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .reconstruction_profile import (
    ReconstructionProfile,
    MaterialProfile,
    SceneProfile,
    SizeProfile,
    MotionProfile,
)


PRESET_NAME_BASELINE = "baseline"
PRESET_NAME_PROFILE_SAFE = "profile_safe"
PRESET_NAME_LOW_TEXTURE_SAFE = "low_texture_safe"
PRESET_NAME_LOW_LIGHT_SAFE = "low_light_safe"
PRESET_NAME_TEXTURE_RETRY_SAFE = "texture_retry_safe"


def _baseline_preset() -> Dict[str, Any]:
    """Always-available fallback — matches what env defaults already do."""
    return {
        "name": PRESET_NAME_BASELINE,
        "colmap": {
            "feature_quality": "high",
            "matcher_type": "exhaustive",
            "max_image_size": 2000,
            "mapper_min_num_matches": 15,
            "patchmatch_resolution_level": 1,
        },
        "openmvs": {
            "texture_resolution": 4096,
            "max_threads": 0,
            "enable_texture_retry": True,
        },
        "rationale": "env defaults; safe everywhere",
    }


def _profile_safe_preset(profile: ReconstructionProfile) -> Dict[str, Any]:
    """
    Generic profile-aware tune.  Conservative deltas only — never crank up
    parameters that could increase memory pressure.
    """
    p = _baseline_preset()
    p["name"] = PRESET_NAME_PROFILE_SAFE
    rationale = ["base from baseline"]

    # Size: large objects need bigger images but lower mesh budget downstream
    if profile.size_profile == SizeProfile.LARGE:
        p["colmap"]["max_image_size"] = 3000
        p["colmap"]["patchmatch_resolution_level"] = 2  # quarter for VRAM safety
        p["openmvs"]["texture_resolution"] = 8192
        rationale.append("size=large → image 3000 + patchmatch L2 + tex 8192")
    elif profile.size_profile == SizeProfile.SMALL:
        p["colmap"]["max_image_size"] = 2000
        p["openmvs"]["texture_resolution"] = 2048
        rationale.append("size=small → image 2000 + tex 2048")
    elif profile.size_profile == SizeProfile.MEDIUM:
        p["colmap"]["max_image_size"] = 2500
        p["openmvs"]["texture_resolution"] = 4096
        rationale.append("size=medium → image 2500 + tex 4096")

    # Material glossy: PatchMatch needs more support → larger window via res_level
    if profile.material_profile == MaterialProfile.GLOSSY:
        p["colmap"]["patchmatch_resolution_level"] = max(
            p["colmap"]["patchmatch_resolution_level"], 1
        )
        rationale.append("material=glossy → keep resolution_level≥1 for stability")

    # Motion fast → tolerate fewer matches, sequential matcher faster on long captures
    if profile.motion_profile == MotionProfile.FAST_MOTION:
        p["colmap"]["matcher_type"] = "sequential"
        p["colmap"]["mapper_min_num_matches"] = 10
        rationale.append("motion=fast → sequential matcher + relaxed min_matches")

    p["rationale"] = "; ".join(rationale)
    return p


def _low_texture_safe_preset(profile: ReconstructionProfile) -> Dict[str, Any]:
    """For LOW_TEXTURE / TRANSPARENT_REFLECTIVE materials."""
    p = _baseline_preset()
    p["name"] = PRESET_NAME_LOW_TEXTURE_SAFE
    p["colmap"]["feature_quality"] = "high"
    p["colmap"]["matcher_type"] = "sequential"
    p["colmap"]["mapper_min_num_matches"] = 8
    # Don't downsample — we need every pixel for SIFT
    p["colmap"]["patchmatch_resolution_level"] = 1
    p["rationale"] = (
        "low_texture/reflective surface — sequential matcher tolerates sparse "
        "feature graph; min_matches relaxed to 8"
    )
    return p


def _low_light_safe_preset(profile: ReconstructionProfile) -> Dict[str, Any]:
    p = _baseline_preset()
    p["name"] = PRESET_NAME_LOW_LIGHT_SAFE
    # Smaller images: noisy images consume PatchMatch VRAM without benefit
    p["colmap"]["max_image_size"] = 1600
    p["colmap"]["matcher_type"] = "sequential"
    p["colmap"]["mapper_min_num_matches"] = 12
    p["colmap"]["patchmatch_resolution_level"] = 2  # half-res; noise mostly
    p["openmvs"]["texture_resolution"] = 2048
    p["rationale"] = (
        "low_light — shrink image to 1600 + half-res PatchMatch to suppress noise; "
        "sequential matcher avoids exhaustive-match noise blowup"
    )
    return p


def _texture_retry_safe_preset(profile: ReconstructionProfile) -> Dict[str, Any]:
    """For OpenMVS TextureMesh crash retries (called by fallback ladder)."""
    p = _baseline_preset()
    p["name"] = PRESET_NAME_TEXTURE_RETRY_SAFE
    p["openmvs"]["texture_resolution"] = 2048
    p["openmvs"]["max_threads"] = 4
    p["openmvs"]["enable_texture_retry"] = False  # avoid retry loops
    p["rationale"] = (
        "OpenMVS TextureMesh crash recovery — clamp atlas to 2048, threads to 4, "
        "no further retry to break loops"
    )
    return p


def resolve_preset(profile: Optional[ReconstructionProfile]) -> Dict[str, Any]:
    """
    Map a profile to a preset.  Order of precedence (most specific first):

        1. material=transparent_reflective | low_texture → low_texture_safe
        2. scene=low_light                              → low_light_safe
        3. anything with at least one known signal      → profile_safe
        4. all unknowns                                 → baseline
    """
    if profile is None:
        return _baseline_preset()

    if profile.material_profile in (MaterialProfile.TRANSPARENT_REFLECTIVE,
                                     MaterialProfile.LOW_TEXTURE):
        return _low_texture_safe_preset(profile)

    if profile.scene_profile == SceneProfile.LOW_LIGHT:
        return _low_light_safe_preset(profile)

    has_known_signal = any(
        v != "unknown" for v in (
            profile.material_profile.value,
            profile.size_profile.value,
            profile.scene_profile.value,
            profile.motion_profile.value,
        )
    )
    if has_known_signal:
        return _profile_safe_preset(profile)

    return _baseline_preset()


def get_preset_by_name(name: str, profile: Optional[ReconstructionProfile] = None) -> Dict[str, Any]:
    """Look up a preset by exact name (used by fallback ladder)."""
    name = (name or "").lower()
    if name == PRESET_NAME_BASELINE:
        return _baseline_preset()
    if name == PRESET_NAME_PROFILE_SAFE:
        return _profile_safe_preset(profile or ReconstructionProfile())
    if name == PRESET_NAME_LOW_TEXTURE_SAFE:
        return _low_texture_safe_preset(profile or ReconstructionProfile())
    if name == PRESET_NAME_LOW_LIGHT_SAFE:
        return _low_light_safe_preset(profile or ReconstructionProfile())
    if name == PRESET_NAME_TEXTURE_RETRY_SAFE:
        return _texture_retry_safe_preset(profile or ReconstructionProfile())
    return _baseline_preset()
