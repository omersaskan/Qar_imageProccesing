"""Sprint 4.5 — reconstruction_command_config tests."""
from __future__ import annotations

import json

import pytest

from modules.reconstruction_engine.reconstruction_command_config import (
    ColmapCommandConfig,
    OpenMVSCommandConfig,
    ReconstructionCommandConfig,
    baseline_command_config,
    from_preset,
)
from modules.reconstruction_engine.reconstruction_preset_resolver import (
    PRESET_NAME_BASELINE,
    PRESET_NAME_LOW_LIGHT_SAFE,
    PRESET_NAME_LOW_TEXTURE_SAFE,
    PRESET_NAME_PROFILE_SAFE,
    get_preset_by_name,
    resolve_preset,
)
from modules.reconstruction_engine.reconstruction_profile import (
    MaterialProfile,
    MotionProfile,
    ReconstructionProfile,
    SceneProfile,
    SizeProfile,
)


def test_baseline_config_has_safe_defaults():
    cfg = baseline_command_config()
    assert cfg.source_preset_name == "baseline"
    assert cfg.colmap.matcher_type == "exhaustive"
    assert cfg.colmap.max_image_size == 2000
    assert cfg.openmvs.texture_resolution == 4096
    assert cfg.openmvs.enable_texture_retry is True


def test_baseline_config_serializable():
    d = baseline_command_config().to_dict()
    json.dumps(d)
    assert d["applied"] is True
    assert "colmap" in d and "openmvs" in d


def test_from_preset_none_falls_back_to_baseline():
    cfg = from_preset(None)
    assert cfg.source_preset_name == "baseline"


def test_from_preset_partial_dict_uses_baseline_for_missing_keys():
    partial = {
        "name": "experimental",
        "colmap": {"matcher_type": "sequential"},
        # openmvs absent
    }
    cfg = from_preset(partial)
    assert cfg.colmap.matcher_type == "sequential"
    # baseline fall-through
    assert cfg.colmap.max_image_size == 2000
    assert cfg.openmvs.texture_resolution == 4096


def test_from_preset_low_texture_safe_routing():
    profile = ReconstructionProfile(material_profile=MaterialProfile.LOW_TEXTURE)
    preset = resolve_preset(profile)
    cfg = from_preset(preset)
    assert cfg.source_preset_name == PRESET_NAME_LOW_TEXTURE_SAFE
    assert cfg.colmap.matcher_type == "sequential"
    assert cfg.colmap.mapper_min_num_matches <= 8


def test_from_preset_low_light_routing():
    profile = ReconstructionProfile(scene_profile=SceneProfile.LOW_LIGHT)
    preset = resolve_preset(profile)
    cfg = from_preset(preset)
    assert cfg.source_preset_name == PRESET_NAME_LOW_LIGHT_SAFE
    assert cfg.colmap.max_image_size <= 1600


def test_from_preset_large_size_bumps_image():
    profile = ReconstructionProfile(size_profile=SizeProfile.LARGE)
    preset = resolve_preset(profile)
    cfg = from_preset(preset)
    assert cfg.source_preset_name == PRESET_NAME_PROFILE_SAFE
    assert cfg.colmap.max_image_size >= 3000
    assert cfg.openmvs.texture_resolution >= 8192


def test_from_preset_garbage_input_safe():
    cfg = from_preset({"colmap": "not a dict", "openmvs": 42})
    # Falls back to baseline values
    assert cfg.colmap.max_image_size == 2000
    assert cfg.openmvs.texture_resolution == 4096
