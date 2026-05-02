"""Sprint 4 — preset_resolver tests."""
from __future__ import annotations

import pytest

from modules.reconstruction_engine.reconstruction_profile import (
    MaterialProfile, MotionProfile, ReconstructionProfile, SceneProfile, SizeProfile,
)
from modules.reconstruction_engine.reconstruction_preset_resolver import (
    PRESET_NAME_BASELINE, PRESET_NAME_LOW_LIGHT_SAFE, PRESET_NAME_LOW_TEXTURE_SAFE,
    PRESET_NAME_PROFILE_SAFE, PRESET_NAME_TEXTURE_RETRY_SAFE,
    get_preset_by_name, resolve_preset,
)


def _make(material=MaterialProfile.UNKNOWN, size=SizeProfile.UNKNOWN,
          scene=SceneProfile.UNKNOWN, motion=MotionProfile.UNKNOWN) -> ReconstructionProfile:
    return ReconstructionProfile(
        material_profile=material, size_profile=size,
        scene_profile=scene, motion_profile=motion,
    )


def test_unknown_profile_returns_baseline():
    p = resolve_preset(_make())
    assert p["name"] == PRESET_NAME_BASELINE


def test_none_profile_returns_baseline():
    p = resolve_preset(None)
    assert p["name"] == PRESET_NAME_BASELINE


def test_low_texture_routes_to_low_texture_safe():
    p = resolve_preset(_make(material=MaterialProfile.LOW_TEXTURE))
    assert p["name"] == PRESET_NAME_LOW_TEXTURE_SAFE
    assert p["colmap"]["matcher_type"] == "sequential"


def test_transparent_reflective_routes_to_low_texture_safe():
    p = resolve_preset(_make(material=MaterialProfile.TRANSPARENT_REFLECTIVE))
    assert p["name"] == PRESET_NAME_LOW_TEXTURE_SAFE


def test_low_light_routes_to_low_light_safe():
    p = resolve_preset(_make(scene=SceneProfile.LOW_LIGHT))
    assert p["name"] == PRESET_NAME_LOW_LIGHT_SAFE
    # Low light: image size shrunk, sequential matcher
    assert p["colmap"]["max_image_size"] <= 1600
    assert p["colmap"]["matcher_type"] == "sequential"


def test_known_signal_routes_to_profile_safe():
    p = resolve_preset(_make(size=SizeProfile.MEDIUM))
    assert p["name"] == PRESET_NAME_PROFILE_SAFE


def test_large_size_bumps_image_and_texture():
    p = resolve_preset(_make(size=SizeProfile.LARGE))
    assert p["colmap"]["max_image_size"] >= 3000
    assert p["openmvs"]["texture_resolution"] >= 8192
    # PatchMatch dropped to half-or-quarter for VRAM safety
    assert p["colmap"]["patchmatch_resolution_level"] >= 2


def test_small_size_keeps_modest_atlas():
    p = resolve_preset(_make(size=SizeProfile.SMALL))
    assert p["openmvs"]["texture_resolution"] <= 2048


def test_glossy_material_keeps_resolution_level():
    p = resolve_preset(_make(material=MaterialProfile.GLOSSY, size=SizeProfile.MEDIUM))
    # patchmatch must NOT be 0 (full-res) — at least 1
    assert p["colmap"]["patchmatch_resolution_level"] >= 1


def test_fast_motion_uses_sequential_matcher():
    p = resolve_preset(_make(motion=MotionProfile.FAST_MOTION, size=SizeProfile.MEDIUM))
    assert p["colmap"]["matcher_type"] == "sequential"
    assert p["colmap"]["mapper_min_num_matches"] <= 12


def test_get_preset_by_name_baseline():
    p = get_preset_by_name(PRESET_NAME_BASELINE)
    assert p["name"] == PRESET_NAME_BASELINE


def test_get_preset_by_name_unknown_falls_back_to_baseline():
    p = get_preset_by_name("not_a_real_preset_name")
    assert p["name"] == PRESET_NAME_BASELINE


def test_texture_retry_safe_caps_threads_and_atlas():
    p = get_preset_by_name(PRESET_NAME_TEXTURE_RETRY_SAFE)
    assert p["openmvs"]["max_threads"] == 4
    assert p["openmvs"]["texture_resolution"] <= 2048
    assert p["openmvs"]["enable_texture_retry"] is False


def test_all_presets_have_required_keys():
    for name in (PRESET_NAME_BASELINE, PRESET_NAME_PROFILE_SAFE,
                 PRESET_NAME_LOW_TEXTURE_SAFE, PRESET_NAME_LOW_LIGHT_SAFE,
                 PRESET_NAME_TEXTURE_RETRY_SAFE):
        p = get_preset_by_name(name)
        assert "colmap" in p and "openmvs" in p and "rationale" in p
        for k in ("feature_quality", "matcher_type", "max_image_size",
                  "mapper_min_num_matches", "patchmatch_resolution_level"):
            assert k in p["colmap"], f"{name} missing colmap.{k}"
        for k in ("texture_resolution", "max_threads", "enable_texture_retry"):
            assert k in p["openmvs"], f"{name} missing openmvs.{k}"
