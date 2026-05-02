"""
Sprint 4.5 — verify ColmapCommandBuilder honors command_config.

These tests exercise the builder with a fake binary path; they don't
invoke COLMAP itself.  We assert command-line shape changes when
command_config differs.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from modules.reconstruction_engine.adapter import ColmapCommandBuilder
from modules.reconstruction_engine.reconstruction_command_config import (
    ColmapCommandConfig,
    ReconstructionCommandConfig,
    baseline_command_config,
    from_preset,
)


def _builder(cfg=None):
    return ColmapCommandBuilder(binary_path="/fake/colmap", use_gpu=False, command_config=cfg)


def test_builder_no_config_uses_legacy_matcher_mode():
    b = _builder()
    cmd = b.matcher("exhaustive", Path("/tmp/db.db"))
    assert "exhaustive_matcher" in cmd


def test_builder_no_config_no_min_num_matches_flag():
    b = _builder()
    cmd = b.mapper(Path("/tmp/db.db"), Path("/tmp/imgs"), Path("/tmp/sparse"))
    flat = " ".join(cmd)
    assert "--Mapper.min_num_matches" not in flat


def test_builder_with_config_sequential_overrides_input_mode():
    cfg = baseline_command_config()
    cfg.colmap.matcher_type = "sequential"
    b = _builder(cfg)
    cmd = b.matcher("exhaustive", Path("/tmp/db.db"))
    assert "sequential_matcher" in cmd


def test_builder_with_config_passes_min_matches():
    cfg = baseline_command_config()
    cfg.colmap.mapper_min_num_matches = 8
    b = _builder(cfg)
    cmd = b.mapper(Path("/tmp/db.db"), Path("/tmp/imgs"), Path("/tmp/sparse"))
    flat = " ".join(cmd)
    assert "--Mapper.min_num_matches" in flat
    assert "8" in cmd


def test_builder_patch_match_includes_resolution_level_cap_when_lvl_ge_1():
    cfg = baseline_command_config()
    cfg.colmap.patchmatch_resolution_level = 2
    b = _builder(cfg)
    cmd = b.patch_match_stereo(Path("/tmp/dense"))
    flat = " ".join(cmd)
    # Quarter-res config should produce a max_image_size cap
    assert "--PatchMatchStereo.max_image_size" in flat


def test_builder_patch_match_no_cap_for_full_res():
    cfg = baseline_command_config()
    cfg.colmap.patchmatch_resolution_level = 0
    b = _builder(cfg)
    cmd = b.patch_match_stereo(Path("/tmp/dense"))
    flat = " ".join(cmd)
    assert "--PatchMatchStereo.max_image_size" not in flat


def test_feature_extractor_camera_model_param_passes_through():
    """Sprint 1 wired the camera model — Sprint 4.5 must not break it."""
    b = _builder()
    cmd = b.feature_extractor(
        Path("/tmp/db.db"), Path("/tmp/imgs"), None,
        max_size=2000, camera_model="OPENCV_FISHEYE",
    )
    assert "OPENCV_FISHEYE" in cmd
