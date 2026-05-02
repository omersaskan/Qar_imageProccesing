"""Sprint 4.6 — runtime fallback loop wiring tests.

These tests drive ReconstructionRunner._run_runtime_fallback_loop and the
runner.run() dispatch directly, using a fake adapter so no real COLMAP /
OpenMVS process runs.  They verify:

  - hardening disabled → legacy loop preserved (runtime loop never invoked)
  - hardening + runtime fallback enabled → preset-driven loop drives attempts
  - failure classes route to the right fallback preset (native_crash, oom,
    missing_file, unknown)
  - missing_file aborts immediately (no retry storm)
  - max_attempts cap respected
  - successful 2nd attempt yields final_status=reconstructed
  - manifest v1.6 fields present
  - adapter cache invalidation rebuilds with new command_config
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from modules.operations import settings as settings_mod
from modules.reconstruction_engine.runner import ReconstructionRunner
from modules.reconstruction_engine.reconstruction_command_config import (
    baseline_command_config,
    from_preset,
)
from modules.reconstruction_engine.reconstruction_preset_resolver import (
    PRESET_NAME_PROFILE_SAFE,
    get_preset_by_name,
)
from modules.reconstruction_engine.failures import (
    InsufficientReconstructionError,
    MissingArtifactError,
    RuntimeReconstructionError,
)
from modules.shared_contracts.models import (
    ReconstructionAudit,
    ReconstructionJob,
)


# ───────────────────────── helpers ─────────────────────────


class FakeAdapter:
    """Stateful adapter; pops a side-effect per run_reconstruction call."""

    engine_type = "colmap_openmvs"
    is_stub = False

    def __init__(self, side_effects: List[Any]):
        self._side_effects = list(side_effects)
        self.calls: List[Dict[str, Any]] = []

    def run_reconstruction(self, frames, exec_dir, density=1.0, enforce_masks=True):
        self.calls.append(
            {
                "frames": list(frames),
                "exec_dir": str(exec_dir),
                "density": density,
                "enforce_masks": enforce_masks,
            }
        )
        if not self._side_effects:
            raise RuntimeReconstructionError("FakeAdapter: no more side effects queued")
        eff = self._side_effects.pop(0)
        if isinstance(eff, BaseException):
            raise eff
        return eff


def _success_results(tmp_path: Path) -> Dict[str, Any]:
    """Minimal results dict the runtime loop accepts."""
    mesh = tmp_path / "fake_mesh.obj"
    mesh.write_text("# fake mesh\n")
    log = tmp_path / "recon.log"
    log.write_text("ok\n")
    return {
        "registered_images": 10,
        "sparse_points": 1000,
        "dense_points_fused": 5000,
        "mesher_used": "poisson",
        "mesh_path": str(mesh),
        "log_path": str(log),
        "texture_path": None,
    }


def _plant_block(runner: ReconstructionRunner, *, profile=None, preset_name="profile_safe"):
    """Plant a v1.6 hardening block on the runner without invoking preflight."""
    runner._effective_settings = None
    profile_dict = profile or {
        "material_profile": "matte",
        "size_profile": "small",
        "scene_profile": "controlled",
        "motion_profile": "stable_orbit",
    }
    preset = get_preset_by_name(PRESET_NAME_PROFILE_SAFE, None)
    preset["name"] = preset_name
    runner._hardening_block = {
        "version": "v1.6",
        "enabled": True,
        "hardening_mode": "runtime_enforced",
        "runtime_fallback_enabled": True,
        "profile": profile_dict,
        "preflight": {"decision": "pass"},
        "preset": preset,
        "active_preset": preset_name,
        "command_config": from_preset(preset).to_dict(),
        "_command_config_obj": from_preset(preset),
        "intrinsics_cache": {"status": "disabled"},
        "attempts": [],
        "fallback_attempts": [],
        "final_attempt": 0,
        "final_status": "pending",
    }


def _make_job(tmp_path: Path) -> ReconstructionJob:
    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True, exist_ok=True)
    return ReconstructionJob(
        job_id="job-4-6",
        capture_session_id="sess-4-6",
        product_id="prod-4-6",
        job_dir=str(job_dir),
        input_frames=[str(tmp_path / "f1.jpg"), str(tmp_path / "f2.jpg"), str(tmp_path / "f3.jpg")],
        source_video_path=None,
    )


# ───────────────────────── classification → routing ─────────────────────────


def test_runtime_loop_first_attempt_uses_initial_preset(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 3)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter([_success_results(tmp_path)])
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    results, idx, engine = runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    assert results is not None
    assert idx == 0
    assert engine == "colmap_openmvs"
    block = runner._hardening_block
    assert block["fallback_attempts"][0]["preset"] == "profile_safe"
    assert block["final_status"] == "reconstructed"


def test_runtime_loop_native_crash_retries_on_low_thread_texture(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 3)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter(
        [
            RuntimeReconstructionError("OpenMVS exit code 3221226505 during TextureMesh"),
            _success_results(tmp_path),
        ]
    )
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    results, idx, _ = runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    assert results is not None
    block = runner._hardening_block
    assert len(block["fallback_attempts"]) == 2
    assert block["fallback_attempts"][0]["failure_class"] == "native_crash"
    assert block["fallback_attempts"][1]["preset"] == "low_thread_texture"
    assert block["fallback_attempts"][1]["status"] == "passed"
    assert block["final_status"] == "reconstructed"
    assert block["active_preset"] == "low_thread_texture"
    assert block["final_attempt"] == 2


def test_runtime_loop_oom_retries_on_safe_low_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 3)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter(
        [
            RuntimeReconstructionError("CUDA error: out of memory"),
            _success_results(tmp_path),
        ]
    )
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    block = runner._hardening_block
    assert block["fallback_attempts"][0]["failure_class"] == "oom"
    assert block["fallback_attempts"][1]["preset"] == "safe_low_resolution"
    assert block["final_status"] == "reconstructed"


def test_runtime_loop_missing_file_aborts_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 5)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter([RuntimeReconstructionError("No such file or directory: dense/fused.ply")])
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    results, idx, _ = runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    assert results is None
    assert idx == -1
    block = runner._hardening_block
    assert len(block["fallback_attempts"]) == 1
    assert block["fallback_attempts"][0]["failure_class"] == "missing_file"
    assert block["fallback_attempts"][0]["next_preset"] is None
    assert block["final_status"] == "failed"
    # Ensure no retry was attempted.
    assert len(fake.calls) == 1


def test_runtime_loop_unknown_failure_uses_default_ladder_step(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 3)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter(
        [
            RuntimeReconstructionError("BA degenerate; sparse model collapsed"),
            _success_results(tmp_path),
        ]
    )
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    block = runner._hardening_block
    assert block["fallback_attempts"][0]["failure_class"] == "unknown"
    # Default ladder picks safe_high_quality after profile_safe.
    assert block["fallback_attempts"][1]["preset"] == "safe_high_quality"
    assert block["final_status"] == "reconstructed"


# ───────────────────────── caps + exhaustion ─────────────────────────


def test_runtime_loop_max_attempts_cap_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 2)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter(
        [
            RuntimeReconstructionError("BA collapsed"),
            RuntimeReconstructionError("still failing"),
            RuntimeReconstructionError("would be 3rd"),
        ]
    )
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    results, idx, _ = runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    assert results is None
    assert len(fake.calls) == 2  # capped, never reaches 3rd
    block = runner._hardening_block
    assert block["final_status"] == "failed"
    assert block["final_attempt"] == 2


def test_runtime_loop_all_attempts_fail_sets_final_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 5)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter([RuntimeReconstructionError("BA collapsed")] * 6)
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    results, idx, _ = runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    assert results is None
    assert runner._hardening_block["final_status"] == "failed"
    assert len(fake.calls) <= 5


def test_runtime_loop_zero_max_attempts_runs_once_then_fails(tmp_path, monkeypatch):
    # Cap floor is 1 so 0 still attempts once (no infinite retry; deterministic).
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 0)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter([RuntimeReconstructionError("BA collapsed")])
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    results, _, _ = runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    assert results is None
    assert len(fake.calls) == 1
    assert runner._hardening_block["final_status"] == "failed"


# ───────────────────────── manifest schema v1.6 ─────────────────────────


def test_manifest_v1_6_attempt_record_has_sprint_4_6_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 3)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter([_success_results(tmp_path)])
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    rec = runner._hardening_block["fallback_attempts"][0]
    for field in (
        "attempt_index",
        "preset_name",
        "command_config",
        "started_at",
        "finished_at",
        "error_summary",
        "next_preset",
    ):
        assert field in rec, f"missing v1.6 field: {field}"
    assert rec["attempt_index"] == 1
    assert rec["preset_name"] == "profile_safe"
    assert rec["command_config"] is not None
    assert rec["status"] == "passed"


def test_manifest_v1_6_attempts_order_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 4)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter(
        [
            RuntimeReconstructionError("OpenMVS exit code 3221226505"),
            RuntimeReconstructionError("CUDA error: out of memory"),
            _success_results(tmp_path),
        ]
    )
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    presets = [a["preset"] for a in runner._hardening_block["fallback_attempts"]]
    assert presets == ["profile_safe", "low_thread_texture", "safe_low_resolution"]
    indices = [a["attempt_index"] for a in runner._hardening_block["fallback_attempts"]]
    assert indices == [1, 2, 3]


def test_manifest_v1_6_serialization_strips_private_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 2)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter([_success_results(tmp_path)])
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    runner._write_hardening_manifest(Path(job.job_dir), runner._hardening_block)
    written = json.loads((Path(job.job_dir) / "reconstruction_hardening.json").read_text(encoding="utf-8"))
    assert "_command_config_obj" not in written
    assert written["version"] == "v1.6"
    assert written["hardening_mode"] == "runtime_enforced"
    assert written["runtime_fallback_enabled"] is True
    assert written["active_preset"] == "profile_safe"


def _stub_upstream_for_hardening(monkeypatch):
    """Stub derive_profile + evaluate_preflight so _run_preset_hardening
    runs without real images / extraction manifest."""
    from modules.reconstruction_engine import (
        reconstruction_profile as profile_mod,
        reconstruction_preflight as preflight_mod,
    )

    fake_profile = profile_mod.ReconstructionProfile()

    class _FakePreflight:
        decision = preflight_mod.PreflightDecision.PASS

        def to_dict(self):
            return {"decision": "pass", "reasons": []}

    monkeypatch.setattr(profile_mod, "derive_profile", lambda **kw: fake_profile)
    monkeypatch.setattr(preflight_mod, "evaluate_preflight", lambda **kw: _FakePreflight())


def test_hardening_block_v1_6_built_with_runtime_enforced_mode(tmp_path, monkeypatch):
    """_run_preset_hardening reflects the runtime_fallback_enabled flag."""
    monkeypatch.setattr(settings_mod.settings, "reconstruction_preset_hardening_enabled", True)
    monkeypatch.setattr(settings_mod.settings, "reconstruction_runtime_fallback_enabled", True)
    _stub_upstream_for_hardening(monkeypatch)

    runner = ReconstructionRunner()
    job = _make_job(tmp_path)
    block = runner._run_preset_hardening(job)
    assert block["version"] == "v1.6"
    assert block["hardening_mode"] == "runtime_enforced"
    assert block["runtime_fallback_enabled"] is True
    assert block["active_preset"] == block["preset"]["name"]


def test_hardening_block_v1_6_manifest_only_when_runtime_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "reconstruction_preset_hardening_enabled", True)
    monkeypatch.setattr(settings_mod.settings, "reconstruction_runtime_fallback_enabled", False)
    _stub_upstream_for_hardening(monkeypatch)
    runner = ReconstructionRunner()
    job = _make_job(tmp_path)
    block = runner._run_preset_hardening(job)
    assert block["hardening_mode"] == "manifest_only"
    assert block["runtime_fallback_enabled"] is False


# ───────────────────────── adapter cache invalidation ─────────────────────────


def test_swap_preset_invalidates_cached_adapters_and_rebuilds_with_new_config(tmp_path, monkeypatch):
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    # Plant stale cached adapters; swap should drop them.
    runner._colmap_cached = "stale-colmap"
    runner._openmvs_cached = "stale-openmvs"

    nxt = runner._swap_to_next_preset("OpenMVS exit code 3221226505")
    assert nxt == "low_thread_texture"
    assert not hasattr(runner, "_colmap_cached")
    assert not hasattr(runner, "_openmvs_cached")
    # New cfg reflected in block
    cfg = runner._hardening_block["_command_config_obj"]
    assert cfg.source_preset_name == "low_thread_texture"


# ───────────────────────── run() dispatch ─────────────────────────


def test_run_dispatch_routes_to_runtime_loop_when_active(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "reconstruction_preset_hardening_enabled", True)
    monkeypatch.setattr(settings_mod.settings, "reconstruction_runtime_fallback_enabled", True)
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 2)

    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    # Skip preflight image reads
    monkeypatch.setattr(runner, "_run_preset_hardening", lambda j: runner._hardening_block)
    monkeypatch.setattr(runner, "_validate_input_frames", lambda f: list(f))
    monkeypatch.setattr(runner, "_finalize_best_attempt", lambda *a, **k: "MANIFEST_OK")

    fake = FakeAdapter([_success_results(tmp_path)])
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    out = runner.run(job)
    assert out == "MANIFEST_OK"
    assert len(fake.calls) == 1


def test_run_dispatch_skips_runtime_loop_when_flag_off(tmp_path, monkeypatch):
    """Hardening on but runtime_fallback off → legacy path runs."""
    monkeypatch.setattr(settings_mod.settings, "reconstruction_preset_hardening_enabled", True)
    monkeypatch.setattr(settings_mod.settings, "reconstruction_runtime_fallback_enabled", False)
    monkeypatch.setattr(settings_mod.settings, "recon_fallback_steps", ["default"])

    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    runner._hardening_block["runtime_fallback_enabled"] = False
    runner._hardening_block["hardening_mode"] = "manifest_only"

    legacy_calls: List[int] = []

    class LegacyFake(FakeAdapter):
        def run_reconstruction(self, frames, exec_dir, density=1.0, enforce_masks=True):
            legacy_calls.append(1)
            return _success_results(tmp_path)

    runner._explicit_adapter = LegacyFake([])
    monkeypatch.setattr(runner, "_run_preset_hardening", lambda j: runner._hardening_block)
    monkeypatch.setattr(runner, "_validate_input_frames", lambda f: list(f))
    monkeypatch.setattr(runner, "_finalize_best_attempt", lambda *a, **k: "LEGACY_OK")

    job = _make_job(tmp_path)
    out = runner.run(job)
    assert out == "LEGACY_OK"
    assert len(legacy_calls) >= 1


def test_run_dispatch_skips_runtime_loop_when_hardening_off(tmp_path, monkeypatch):
    """Hardening flag itself off → block is None, legacy path runs verbatim."""
    monkeypatch.setattr(settings_mod.settings, "reconstruction_preset_hardening_enabled", False)
    monkeypatch.setattr(settings_mod.settings, "reconstruction_runtime_fallback_enabled", True)
    monkeypatch.setattr(settings_mod.settings, "recon_fallback_steps", ["default"])

    runner = ReconstructionRunner()
    fake = FakeAdapter([_success_results(tmp_path)])
    runner._explicit_adapter = fake

    monkeypatch.setattr(runner, "_validate_input_frames", lambda f: list(f))
    monkeypatch.setattr(runner, "_finalize_best_attempt", lambda *a, **k: "LEGACY_OK")

    job = _make_job(tmp_path)
    out = runner.run(job)
    assert out == "LEGACY_OK"
    assert runner._hardening_block is None


def test_run_preflight_reject_writes_capture_quality_rejected_and_skips_recon(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "reconstruction_preset_hardening_enabled", True)
    monkeypatch.setattr(settings_mod.settings, "reconstruction_runtime_fallback_enabled", True)

    runner = ReconstructionRunner()
    rejected_block = {
        "version": "v1.6",
        "enabled": True,
        "hardening_mode": "runtime_enforced",
        "runtime_fallback_enabled": True,
        "profile": {},
        "preflight": {"decision": "reject", "reasons": ["too few frames"]},
        "preset": {"name": "profile_safe"},
        "active_preset": "profile_safe",
        "command_config": baseline_command_config().to_dict(),
        "_command_config_obj": baseline_command_config(),
        "intrinsics_cache": {"status": "disabled"},
        "attempts": [],
        "fallback_attempts": [],
        "final_attempt": 0,
        "final_status": "pending",
    }
    monkeypatch.setattr(runner, "_run_preset_hardening", lambda j: rejected_block)

    fake = FakeAdapter([_success_results(tmp_path)])
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    with pytest.raises(Exception):
        runner.run(job)
    # Ensure the adapter never ran.
    assert len(fake.calls) == 0
    # Hardening manifest reflects rejection
    hardening = json.loads((Path(job.job_dir) / "reconstruction_hardening.json").read_text(encoding="utf-8"))
    assert hardening["final_status"] == "capture_quality_rejected"
    audit = json.loads((Path(job.job_dir) / "reconstruction_audit.json").read_text(encoding="utf-8"))
    assert audit["final_status"] == "capture_quality_rejected"


# ───────────────────────── audit + side-effects ─────────────────────────


def test_runtime_loop_records_audit_attempts_for_each_iteration(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 3)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter(
        [
            RuntimeReconstructionError("OpenMVS exit code 3221226505"),
            _success_results(tmp_path),
        ]
    )
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    assert len(audit.attempts) == 2
    assert audit.attempts[0].status == "failed"
    assert audit.attempts[1].status == "success"
    assert audit.attempts[0].metadata["preset"] == "profile_safe"
    assert audit.attempts[1].metadata["preset"] == "low_thread_texture"


def test_runtime_loop_creates_attempt_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 2)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter([_success_results(tmp_path)])
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    assert (Path(job.job_dir) / "attempt_1_profile_safe").is_dir()


def test_runtime_loop_no_infinite_retry_on_persistent_native_crash(tmp_path, monkeypatch):
    """Even if every attempt native-crashes, max_attempts caps the loop."""
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 3)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter([RuntimeReconstructionError("OpenMVS exit code 3221226505")] * 10)
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    results, _, _ = runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    assert results is None
    assert len(fake.calls) <= 3
    assert runner._hardening_block["final_status"] == "failed"


def test_runtime_loop_command_config_snapshot_changes_per_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 3)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter(
        [
            RuntimeReconstructionError("OpenMVS exit code 3221226505"),
            _success_results(tmp_path),
        ]
    )
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    snaps = [a["command_config"] for a in runner._hardening_block["fallback_attempts"]]
    assert snaps[0]["source_preset_name"] == "profile_safe"
    assert snaps[1]["source_preset_name"] == "low_thread_texture"


def test_runtime_loop_started_finished_timestamps_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 2)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter([_success_results(tmp_path)])
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    rec = runner._hardening_block["fallback_attempts"][0]
    assert rec["started_at"] and rec["finished_at"]
    assert rec["finished_at"] >= rec["started_at"]


def test_runtime_loop_attempts_alias_mirrors_fallback_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "fallback_ladder_max_attempts", 2)
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    fake = FakeAdapter([_success_results(tmp_path)])
    runner._explicit_adapter = fake

    job = _make_job(tmp_path)
    audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
    runner._run_runtime_fallback_loop(job, ["a", "b", "c"], audit, Path(job.job_dir))

    block = runner._hardening_block
    assert len(block["attempts"]) == len(block["fallback_attempts"]) == 1


def test_record_fallback_attempt_active_preset_updates_on_pass(tmp_path):
    runner = ReconstructionRunner()
    _plant_block(runner, preset_name="profile_safe")
    runner._record_fallback_attempt(
        attempt_num=1, preset_name="low_thread_texture", status="passed"
    )
    assert runner._hardening_block["active_preset"] == "low_thread_texture"


def test_runtime_fallback_active_false_when_block_missing():
    runner = ReconstructionRunner()
    runner._hardening_block = None
    assert runner._runtime_fallback_active() is False


def test_runtime_fallback_active_false_when_flag_off(monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "reconstruction_runtime_fallback_enabled", False)
    runner = ReconstructionRunner()
    _plant_block(runner)
    assert runner._runtime_fallback_active() is False


def test_runtime_fallback_active_true_when_both_set(monkeypatch):
    monkeypatch.setattr(settings_mod.settings, "reconstruction_runtime_fallback_enabled", True)
    runner = ReconstructionRunner()
    _plant_block(runner)
    assert runner._runtime_fallback_active() is True


def test_peek_next_preset_returns_none_without_block():
    runner = ReconstructionRunner()
    runner._hardening_block = None
    assert runner._peek_next_preset("any error") is None


def test_peek_next_preset_routes_native_crash():
    runner = ReconstructionRunner()
    _plant_block(runner)
    nxt = runner._peek_next_preset("OpenMVS exit code 3221226505")
    assert nxt == "low_thread_texture"
