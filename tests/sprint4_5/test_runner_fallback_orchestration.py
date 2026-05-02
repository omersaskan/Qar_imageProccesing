"""
Sprint 4.5 — runner-level fallback orchestration tests.

Without invoking real COLMAP, we drive the runner's bookkeeping helpers
(_record_fallback_attempt, _swap_to_next_preset) directly and verify:

  - hardening block accumulates attempts in order
  - native_crash routes to low_thread_texture
  - oom routes to safe_low_resolution
  - missing_file aborts (returns None)
  - max_attempts cap enforced (no infinite retry)
  - successful 2nd attempt sets final_status=reconstructed
  - non-serializable cached object stripped on disk write
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.reconstruction_engine.runner import ReconstructionRunner


def _make_runner_with_block(profile_dict=None):
    """Build a runner + plant a synthetic hardening block (no real job)."""
    from modules.reconstruction_engine.reconstruction_command_config import baseline_command_config

    runner = ReconstructionRunner()
    runner._effective_settings = None
    runner._hardening_block = {
        "version": "v1.5",
        "enabled": True,
        "profile": profile_dict or {
            "material_profile": "matte",
            "size_profile": "small",
            "scene_profile": "controlled",
            "motion_profile": "stable_orbit",
        },
        "preflight": {"decision": "pass"},
        "preset": {"name": "profile_safe"},
        "command_config": baseline_command_config().to_dict(),
        "_command_config_obj": baseline_command_config(),
        "intrinsics_cache": {"status": "disabled"},
        "fallback_attempts": [],
        "final_attempt": 0,
        "final_status": "pending",
    }
    return runner


def test_record_attempt_first_failure_sets_retrying():
    runner = _make_runner_with_block()
    runner._record_fallback_attempt(
        attempt_num=1, preset_name="profile_safe", status="failed",
        failure_class="native_crash", exit_code=3221226505,
        next_action="low_thread_texture",
    )
    block = runner._hardening_block
    assert len(block["fallback_attempts"]) == 1
    assert block["fallback_attempts"][0]["preset"] == "profile_safe"
    assert block["fallback_attempts"][0]["next_action"] == "low_thread_texture"
    assert block["final_status"] == "retrying"
    assert block["final_attempt"] == 1


def test_record_attempt_pass_sets_final_reconstructed():
    runner = _make_runner_with_block()
    runner._record_fallback_attempt(
        attempt_num=1, preset_name="profile_safe", status="passed",
    )
    assert runner._hardening_block["final_status"] == "reconstructed"


def test_record_attempt_fail_no_next_action_sets_failed():
    runner = _make_runner_with_block()
    runner._record_fallback_attempt(
        attempt_num=2, preset_name="baseline", status="failed",
        failure_class="unknown", next_action=None,
    )
    assert runner._hardening_block["final_status"] == "failed"


def test_swap_to_next_preset_native_crash_jumps_to_low_thread_texture():
    runner = _make_runner_with_block()
    # Plant a baseline cached adapter to verify it gets invalidated
    runner._colmap_cached = "stale"
    nxt = runner._swap_to_next_preset(error_excerpt="OpenMVS exit code 3221226505")
    assert nxt == "low_thread_texture"
    block = runner._hardening_block
    assert block["preset"]["name"] == "low_thread_texture"
    assert block["_command_config_obj"].source_preset_name == "low_thread_texture"
    # Cached adapter dropped
    assert not hasattr(runner, "_colmap_cached")


def test_swap_to_next_preset_oom_jumps_to_safe_low_resolution():
    runner = _make_runner_with_block()
    nxt = runner._swap_to_next_preset(error_excerpt="CUDA error: out of memory")
    assert nxt == "safe_low_resolution"
    cfg_obj = runner._hardening_block["_command_config_obj"]
    assert cfg_obj.colmap.max_image_size <= 1500   # half-size OOM recovery


def test_swap_to_next_preset_missing_file_returns_none():
    runner = _make_runner_with_block()
    nxt = runner._swap_to_next_preset(error_excerpt="No such file or directory: /missing")
    assert nxt is None
    assert runner._hardening_block["final_status"] == "failed"


def test_swap_respects_max_attempts(monkeypatch):
    runner = _make_runner_with_block()
    # Cap at 2 attempts; pre-load 2 attempts → next swap returns None
    from modules.operations import settings as settings_mod
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 2)
    runner._record_fallback_attempt(1, "profile_safe", "failed", "runtime", next_action="safe_high_quality")
    runner._record_fallback_attempt(2, "safe_high_quality", "failed", "runtime", next_action="safe_low_resolution")
    nxt = runner._swap_to_next_preset(error_excerpt="another runtime error")
    assert nxt is None
    assert runner._hardening_block["final_status"] == "failed"


def test_swap_no_block_returns_none():
    runner = ReconstructionRunner()
    runner._hardening_block = None
    assert runner._swap_to_next_preset("anything") is None


def test_record_no_block_silent_noop():
    runner = ReconstructionRunner()
    runner._hardening_block = None
    runner._record_fallback_attempt(1, "x", "failed")  # does not raise


def test_hardening_block_serialization_strips_private_keys(tmp_path):
    runner = _make_runner_with_block()
    runner._record_fallback_attempt(1, "profile_safe", "passed")
    runner._write_hardening_manifest(tmp_path, runner._hardening_block)
    out_path = tmp_path / "reconstruction_hardening.json"
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    # Cached object should NOT be in the on-disk JSON
    assert "_command_config_obj" not in written
    assert written["final_status"] == "reconstructed"


def test_full_attempt_sequence_native_crash_then_pass():
    """End-to-end: 1st attempt native crash → swap to low_thread_texture → pass."""
    runner = _make_runner_with_block()
    # Attempt 1 fails with native crash
    runner._record_fallback_attempt(
        1, "profile_safe", "failed",
        failure_class="native_crash", exit_code=3221226505,
        next_action="low_thread_texture",
        error_excerpt="OpenMVS exit code 3221226505",
    )
    # Runner picks the next preset
    nxt = runner._swap_to_next_preset("OpenMVS exit code 3221226505")
    assert nxt == "low_thread_texture"
    # Attempt 2 passes
    runner._record_fallback_attempt(2, "low_thread_texture", "passed")
    block = runner._hardening_block
    assert block["final_attempt"] == 2
    assert block["final_status"] == "reconstructed"
    assert len(block["fallback_attempts"]) == 2
    assert block["fallback_attempts"][0]["status"] == "failed"
    assert block["fallback_attempts"][1]["status"] == "passed"


def test_zero_max_attempts_blocks_swap(monkeypatch):
    runner = _make_runner_with_block()
    from modules.operations import settings as settings_mod
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 0)
    nxt = runner._swap_to_next_preset("any error")
    assert nxt is None
