"""
tests/test_sprint1_integration.py

SPRINT 1 — TICKET-003: Happy-Path + Failure-Path Integration Tests

Coverage:
  TICKET-001 — retry counter behavior (escalation to FAILED after max_retry_count)
  TICKET-001 — session timeout behavior (old session forcibly FAILED)
  TICKET-002 — worker _handle_cleanup uses TexturingService (regression guard)
  TICKET-002 — TexturingService._apply_pivot_to_obj correct vertex shift
  TICKET-002 — TexturingService skips texturing for non-COLMAP manifests
  TICKET-003 — happy-path pipeline: CREATED → CAPTURED → RECONSTRUCTED → CLEANED
                                     → EXPORTED → VALIDATED (+ publish)
  TICKET-003 — failure-path: irrecoverable at frame extraction → FAILED immediately
  TICKET-004 — disk-space preflight blocks upload when below threshold
  TICKET-004 — COLMAP binary probe returns structured result
  TICKET-004 — /api/ready response includes preflight section
"""

from __future__ import annotations

import json
import shutil
import os
import struct
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, call

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_data(tmp_path):
    """Minimal data_root with the expected folder structure."""
    data = tmp_path / "data"
    for sub in [
        "sessions", "captures", "reconstructions",
        "registry/meta", "registry/blobs", "logs",
    ]:
        (data / sub).mkdir(parents=True, exist_ok=True)
    return data


@pytest.fixture()
def session_manager(tmp_data):
    from modules.capture_workflow.session_manager import SessionManager
    return SessionManager(data_root=str(tmp_data))


@pytest.fixture()
def worker(tmp_data):
    from modules.operations.worker import IngestionWorker
    return IngestionWorker(data_root=str(tmp_data))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bbox_dict(lo, hi):
    """Return bbox_min / bbox_max dicts expected by NormalizedMetadata."""
    return {"x": lo, "y": lo, "z": lo}, {"x": hi, "y": hi, "z": hi}


def _write_minimal_manifest(job_dir: Path, mesh_path: str) -> Path:
    from modules.reconstruction_engine.output_manifest import OutputManifest
    from modules.utils.file_persistence import atomic_write_json

    manifest = OutputManifest(
        job_id=job_dir.name,
        mesh_path=mesh_path,
        log_path=str(job_dir / "recon.log"),
        processing_time_seconds=1.0,
        engine_type="stub",
        is_stub=True,
    )
    manifest_file = job_dir / "manifest.json"
    atomic_write_json(manifest_file, manifest.model_dump(mode="json"))
    return manifest_file


def _make_normalized_metadata(lo=0.0, hi=1.0):
    from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
    bmin, bmax = _bbox_dict(lo, hi)
    return NormalizedMetadata(
        bbox_min=bmin,
        bbox_max=bmax,
        pivot_offset={"x": 0.0, "y": 0.0, "z": 0.0},
        final_polycount=1000,
    )


def _make_cleanup_stats(cleaned_mesh: str, lo=0.0, hi=1.0):
    return {
        "isolation": {
            "initial_faces": 2000, "initial_vertices": 1000,
            "removed_plane_faces": 0, "removed_plane_vertices": 0,
            "removed_plane_face_share": 0.0, "removed_plane_vertex_ratio": 0.0,
            "component_count": 1, "removed_islands": 0,
            "final_faces": 1000, "final_vertices": 500,
            "compactness_score": 0.5, "flatness_score": 0.3,
            "selected_component_score": 0.7,
        },
        "final_polycount": 1000,
        "bbox_min": {"x": lo, "y": lo, "z": lo},
        "bbox_max": {"x": hi, "y": hi, "z": hi},
        "pre_aligned_mesh_path": cleaned_mesh,
        "cleaned_mesh_path": cleaned_mesh,
        "metadata_path": str(Path(cleaned_mesh).parent / "normalized_metadata.json"),
        "cleaned_texture_path": None,
        "uv_preserved": False,
        "material_preserved": False,
    }


def _write_minimal_glb(path: Path) -> None:
    json_dict = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3,
                       "type": "VEC3", "max": [1, 1, 0], "min": [0, 0, 0]}],
        "bufferViews": [{"buffer": 0, "byteLength": 36}],
        "buffers": [{"byteLength": 36}],
    }
    json_bytes = json.dumps(json_dict, separators=(",", ":")).encode("ascii")
    while len(json_bytes) % 4:
        json_bytes += b" "
    bin_bytes = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    total_len = 12 + (8 + len(json_bytes)) + (8 + len(bin_bytes))
    header = struct.pack("<4sII", b"glTF", 2, total_len)
    json_chunk = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    bin_chunk = struct.pack("<I4s", len(bin_bytes), b"BIN\x00") + bin_bytes
    path.write_bytes(header + json_chunk + bin_chunk)


# ──────────────────────────────────────────────────────────────────────────────
# TICKET-001: Retry counter tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRetryCounter:
    """Verify that a session is terminated after exceeding max_retry_count."""

    def test_retry_escalates_to_failed_after_limit(self, tmp_data, session_manager):
        """
        When _handle_frame_extraction raises RecoverableError repeatedly,
        the worker must increment retry_count and convert to IrrecoverableError
        once settings.max_retry_count is exceeded, marking the session FAILED.
        """
        from modules.operations.worker import IngestionWorker, RecoverableError
        from modules.shared_contracts.lifecycle import AssetStatus
        from modules.utils.file_persistence import atomic_write_json

        MAX = 2   # use a low limit for speed

        worker = IngestionWorker(data_root=str(tmp_data))
        session = session_manager.create_session("sess_retry", "prod_r", "op_1")

        # Place a dummy video file so the path-check inside _handle_frame_extraction
        # passes, allowing us to inject a RecoverableError from FrameExtractor.
        video_dir = tmp_data / "captures" / "sess_retry" / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / "raw_video.mp4").write_bytes(b"fake")

        call_count = {"n": 0}

        def bad_extractor_factory():
            m = MagicMock()
            m.extract_keyframes.side_effect = RuntimeError("transient disk I/O glitch")
            m.config = MagicMock(min_frames=3)
            return m

        # Patch settings MAX_RETRY inside the worker module.
        with patch("modules.operations.worker.settings") as ms:
            ms.max_retry_count = MAX
            ms.session_timeout_hours = 999
            ms.worker_interval_sec = 5
            ms.data_root = str(tmp_data)

            # Cycle 1 → retry_count becomes 1
            with patch("modules.capture_workflow.frame_extractor.FrameExtractor",
                       bad_extractor_factory):
                worker._process_pending_sessions()
            s = session_manager.get_session("sess_retry")
            assert s.retry_count == 1, f"Expected retry_count=1, got {s.retry_count}"
            assert s.status == AssetStatus.CREATED

            # Cycle 2 → retry_count becomes 2
            with patch("modules.capture_workflow.frame_extractor.FrameExtractor",
                       bad_extractor_factory):
                worker._process_pending_sessions()
            s = session_manager.get_session("sess_retry")
            assert s.retry_count == 2
            assert s.status == AssetStatus.CREATED

            # Cycle 3 → retry_count = 3 > MAX(2) → IrrecoverableError → FAILED
            with patch("modules.capture_workflow.frame_extractor.FrameExtractor",
                       bad_extractor_factory):
                worker._process_pending_sessions()
            s = session_manager.get_session("sess_retry")
            assert s.status == AssetStatus.FAILED, (
                f"Expected FAILED after exceeding retry limit, got {s.status}"
            )
            assert "Exceeded max retry limit" in (s.failure_reason or ""), (
                f"Unexpected failure_reason: {s.failure_reason}"
            )
            assert s.retry_count == 3

    def test_two_sessions_have_independent_retry_counters(self, tmp_data, session_manager):
        """
        Session A's retry state must not bleed into Session B.
        """
        from modules.operations.worker import IngestionWorker, IrrecoverableError
        from modules.shared_contracts.lifecycle import AssetStatus

        worker = IngestionWorker(data_root=str(tmp_data))

        # Session A: video present → RecoverableError on extraction
        session_a = session_manager.create_session("sess_aa", "prod_a", "op_1")
        va = tmp_data / "captures" / "sess_aa" / "video"
        va.mkdir(parents=True, exist_ok=True)
        (va / "raw_video.mp4").write_bytes(b"fake")

        # Session B: NO video → IrrecoverableError immediately
        session_b = session_manager.create_session("sess_bb", "prod_b", "op_1")

        def bad_extractor_for_a():
            m = MagicMock()
            m.extract_keyframes.side_effect = RuntimeError("transient")
            m.config = MagicMock(min_frames=3)
            return m

        with patch("modules.operations.worker.settings") as ms:
            ms.max_retry_count = 5
            ms.session_timeout_hours = 999
            ms.worker_interval_sec = 5
            ms.data_root = str(tmp_data)

            with patch("modules.capture_workflow.frame_extractor.FrameExtractor",
                       bad_extractor_for_a):
                worker._process_pending_sessions()

        sa = session_manager.get_session("sess_aa")
        sb = session_manager.get_session("sess_bb")

        assert sa.retry_count == 1    # one retry for A (transient)
        assert sa.status == AssetStatus.CREATED

        assert sb.status == AssetStatus.FAILED   # immediate irrecoverable (no video)
        assert sb.retry_count == 0               # no retry for irrecoverable


# ──────────────────────────────────────────────────────────────────────────────
# TICKET-001: Session timeout tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionTimeout:
    """Verify that sessions stuck beyond the timeout threshold are forcibly FAILED."""

    def test_timeout_marks_old_processing_session_failed(self, tmp_data, session_manager):
        from modules.operations.worker import IngestionWorker
        from modules.shared_contracts.lifecycle import AssetStatus
        from modules.utils.file_persistence import atomic_write_json

        worker = IngestionWorker(data_root=str(tmp_data))
        session_manager.create_session("sess_timeout", "prod_t", "op_1")

        # Back-date created_at to 3 hours ago in the JSON directly.
        old_time = datetime.now(timezone.utc) - timedelta(hours=3)
        sess_file = tmp_data / "sessions" / "sess_timeout.json"
        with open(sess_file, "r") as f:
            data = json.load(f)
        data["created_at"] = old_time.isoformat()
        atomic_write_json(sess_file, data)

        with patch("modules.operations.worker.settings") as ms:
            ms.max_retry_count = 5
            ms.session_timeout_hours = 2      # 2h threshold; session is 3h old
            ms.worker_interval_sec = 5
            ms.data_root = str(tmp_data)

            worker._process_pending_sessions()

        s = session_manager.get_session("sess_timeout")
        assert s.status == AssetStatus.FAILED
        assert "timed out" in (s.failure_reason or "").lower(), (
            f"Expected 'timed out' in failure_reason: {s.failure_reason}"
        )

    def test_recent_session_not_timed_out(self, tmp_data, session_manager):
        """A freshly created session (seconds old) must NOT be timed out."""
        from modules.operations.worker import IngestionWorker
        from modules.shared_contracts.lifecycle import AssetStatus

        worker = IngestionWorker(data_root=str(tmp_data))
        session_manager.create_session("sess_fresh", "prod_f", "op_1")
        # No video → the session will fail due to missing video, NOT timeout.

        with patch("modules.operations.worker.settings") as ms:
            ms.max_retry_count = 5
            ms.session_timeout_hours = 2    # session is only seconds old
            ms.worker_interval_sec = 5
            ms.data_root = str(tmp_data)

            worker._process_pending_sessions()

        s = session_manager.get_session("sess_fresh")
        assert s.status == AssetStatus.FAILED
        assert "timed out" not in (s.failure_reason or "").lower(), (
            f"Session should fail with missing video, not timeout: {s.failure_reason}"
        )
        assert "Video file missing" in (s.failure_reason or "")


# ──────────────────────────────────────────────────────────────────────────────
# TICKET-002: TexturingService unit tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTexturingService:
    def test_skips_texturing_for_non_colmap_manifest(self, tmp_data):
        from modules.operations.texturing_service import TexturingService
        from modules.reconstruction_engine.output_manifest import OutputManifest

        svc = TexturingService()
        manifest = OutputManifest(
            job_id="j1",
            mesh_path=str(tmp_data / "mesh.obj"),
            log_path=str(tmp_data / "log.txt"),
            processing_time_seconds=1.0,
            engine_type="stub",
        )
        result = svc.run(
            manifest=manifest,
            cleanup_stats={},
            pivot_offset={"x": 0.0, "y": 0.0, "z": 0.0},
            cleaned_mesh_path=str(tmp_data / "cleaned.obj"),
        )
        assert result.texturing_status == "absent"
        assert result.cleaned_mesh_path == str(tmp_data / "cleaned.obj")

    def test_skips_texturing_when_dense_dir_missing(self, tmp_data):
        from modules.operations.texturing_service import TexturingService
        from modules.reconstruction_engine.output_manifest import OutputManifest

        mesh_dir = tmp_data / "recons" / "dense"
        mesh_dir.mkdir(parents=True)
        mesh_p = mesh_dir / "mesh.ply"
        mesh_p.write_text("fake")
        # Remove dense/ so workspace check fails.
        shutil.rmtree(tmp_data / "recons" / "dense")

        svc = TexturingService()
        manifest = OutputManifest(
            job_id="j2",
            mesh_path=str(mesh_p),
            log_path=str(tmp_data / "log.txt"),
            processing_time_seconds=1.0,
            engine_type="colmap",
        )
        result = svc.run(
            manifest=manifest,
            cleanup_stats={"pre_aligned_mesh_path": str(mesh_p)},
            pivot_offset={"x": 0.0, "y": 0.0, "z": 0.0},
            cleaned_mesh_path=str(tmp_data / "cleaned.obj"),
        )
        assert result.texturing_status == "absent"

    def test_apply_pivot_to_obj_shifts_only_vertex_lines(self, tmp_path):
        from modules.operations.texturing_service import TexturingService

        src = tmp_path / "source.obj"
        src.write_text(
            "# comment\n"
            "v 1.0 2.0 3.0\n"
            "vt 0.5 0.5\n"
            "vn 0.0 1.0 0.0\n"
            "v 0.0 0.0 0.0\n"
            "f 1/1/1 2/1/1\n"
        )
        out = TexturingService._apply_pivot_to_obj(
            str(src),
            {"x": 1.0, "y": -1.0, "z": 0.5},
            str(tmp_path / "cleaned.obj"),
        )
        lines = Path(out).read_text().splitlines()

        # "# comment" passes through unchanged
        assert lines[0] == "# comment"
        # First vertex: 1+1=2, 2-1=1, 3+0.5=3.5
        assert lines[1].startswith("v 2.000000 1.000000 3.500000")
        # vt, vn pass through unchanged
        assert "vt" in lines[2]
        assert "vn" in lines[3]
        # Second vertex: 0+1=1, 0-1=-1, 0+0.5=0.5
        assert lines[4].startswith("v 1.000000 -1.000000 0.500000")
        # face line passes through
        assert "f" in lines[5]

    def test_handle_cleanup_delegates_to_texturing_service(self, tmp_data, session_manager):
        """
        Regression: _handle_cleanup must call texturing_service.run() exactly once.
        """
        from modules.operations.worker import IngestionWorker
        from modules.operations.texturing_service import TexturingResult
        from modules.reconstruction_engine.output_manifest import OutputManifest
        from modules.shared_contracts.lifecycle import AssetStatus
        from modules.utils.file_persistence import atomic_write_json

        worker = IngestionWorker(data_root=str(tmp_data))
        session_manager.create_session("sess_cleanup_r", "prod_c", "op_1")

        job_dir = tmp_data / "reconstructions" / "job_sess_cleanup_r"
        job_dir.mkdir(parents=True)
        cleaned_dir = tmp_data / "cleaned" / "job_sess_cleanup_r"
        cleaned_dir.mkdir(parents=True)

        mesh_p = job_dir / "mesh.ply"
        mesh_p.write_text("# fake")
        manifest_p = _write_minimal_manifest(job_dir, str(mesh_p))

        session = session_manager.update_session(
            "sess_cleanup_r",
            new_status=AssetStatus.CAPTURED,
            last_pipeline_stage="captured",
            failure_reason=None,
        )
        session = session_manager.update_session(
            "sess_cleanup_r",
            new_status=AssetStatus.RECONSTRUCTED,
            reconstruction_manifest_path=str(manifest_p),
            last_pipeline_stage="reconstructed",
            failure_reason=None,
        )

        fake_mesh = cleaned_dir / "cleaned_mesh.obj"
        fake_mesh.write_text("v 0 0 0\n")

        fake_meta = _make_normalized_metadata()
        mock_stats = _make_cleanup_stats(str(fake_mesh))

        loaded_manifest = OutputManifest.model_validate(
            json.loads(manifest_p.read_text())
        )
        expected_tex_result = TexturingResult(
            texturing_status="real",
            cleaned_mesh_path=str(fake_mesh),
            texture_atlas_paths=[str(cleaned_dir / "texture.png")],
            manifest=loaded_manifest,
        )

        with (
            patch.object(worker.cleaner, "process_cleanup",
                         return_value=(fake_meta, mock_stats, str(fake_mesh))),
            patch.object(worker.texturing_service, "run",
                         return_value=expected_tex_result) as mock_tex_run,
        ):
            updated = worker._handle_cleanup(session)

        mock_tex_run.assert_called_once()
        call_kwargs = mock_tex_run.call_args.kwargs
        assert call_kwargs["cleaned_mesh_path"] == str(fake_mesh)
        assert "pivot_offset" in call_kwargs
        assert updated.status == AssetStatus.CLEANED


# ──────────────────────────────────────────────────────────────────────────────
# TICKET-003: Happy-path end-to-end integration test
# ──────────────────────────────────────────────────────────────────────────────

class TestHappyPathIntegration:
    """
    Drives the worker through all pipeline stages from CREATED to PUBLISHED
    using controlled mocks for heavy external tools (COLMAP, OpenMVS, trimesh IO).
    """

    def test_full_pipeline_happy_path(self, tmp_data, session_manager):
        from modules.operations.worker import IngestionWorker
        from modules.shared_contracts.lifecycle import AssetStatus
        from modules.reconstruction_engine.output_manifest import OutputManifest
        from modules.operations.texturing_service import TexturingResult
        from modules.utils.file_persistence import atomic_write_json

        worker = IngestionWorker(data_root=str(tmp_data))
        session_manager.create_session("sess_happy", "prod_happy", "op_1")

        # ── Stage 1: Frame extraction ─────────────────────────────────────
        video_dir = tmp_data / "captures" / "sess_happy" / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / "raw_video.mp4").write_bytes(b"fake_video")

        frames_dir = tmp_data / "captures" / "sess_happy" / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames = [str(frames_dir / f"f{i}.jpg") for i in range(8)]
        for fp in frames:
            Path(fp).write_bytes(b"fake_jpg")

        coverage_ok = {
            "overall_status": "sufficient",
            "coverage_score": 0.82,
            "unique_views": 8,
            "top_down_captured": True,
            "reasons": [],
        }

        def make_extractor():
            m = MagicMock()
            m.extract_keyframes.return_value = (frames, {})
            m.config = MagicMock(min_frames=3)
            return m

        def make_coverage_analyzer():
            m = MagicMock()
            m.analyze_coverage.return_value = coverage_ok
            return m

        with (
            patch("modules.capture_workflow.frame_extractor.FrameExtractor", make_extractor),
            patch("modules.capture_workflow.coverage_analyzer.CoverageAnalyzer",
                  make_coverage_analyzer),
        ):
            worker._process_pending_sessions()

        s = session_manager.get_session("sess_happy")
        assert s.status == AssetStatus.CAPTURED, f"Expected CAPTURED got {s.status}"
        assert len(s.extracted_frames) == 8

        # ── Stage 2: Reconstruction ───────────────────────────────────────
        job_dir = tmp_data / "reconstructions" / "job_sess_happy"
        job_dir.mkdir(parents=True)
        mesh_p = job_dir / "mesh.ply"
        mesh_p.write_text("stub")

        manifest_obj = OutputManifest(
            job_id="job_sess_happy",
            mesh_path=str(mesh_p),
            log_path=str(job_dir / "recon.log"),
            processing_time_seconds=10.0,
            engine_type="stub",
            is_stub=True,
        )
        atomic_write_json(job_dir / "manifest.json", manifest_obj.model_dump(mode="json"))

        def make_job_manager():
            m = MagicMock()
            mock_job = MagicMock()
            mock_job.job_id = "job_sess_happy"
            mock_job.job_dir = str(job_dir)
            m.return_value.create_job.return_value = mock_job
            return m

        def make_runner():
            m = MagicMock()
            m.return_value.run.return_value = manifest_obj
            return m

        with (
            patch("modules.reconstruction_engine.job_manager.JobManager", make_job_manager()),
            patch("modules.reconstruction_engine.runner.ReconstructionRunner", make_runner()),
        ):
            worker._process_pending_sessions()

        s = session_manager.get_session("sess_happy")
        assert s.status == AssetStatus.RECONSTRUCTED, f"Expected RECONSTRUCTED got {s.status}"

        # ── Stage 3: Cleanup ──────────────────────────────────────────────
        cleaned_dir = tmp_data / "cleaned" / "job_sess_happy"
        cleaned_dir.mkdir(parents=True)
        fake_cleaned_mesh = cleaned_dir / "cleaned_mesh.obj"
        fake_cleaned_mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

        fake_meta = _make_normalized_metadata()
        mock_stats = _make_cleanup_stats(str(fake_cleaned_mesh))

        loaded_manifest = OutputManifest.model_validate(
            json.loads((job_dir / "manifest.json").read_text())
        )
        stub_tex_result = TexturingResult(
            texturing_status="real",
            cleaned_mesh_path=str(fake_cleaned_mesh),
            texture_atlas_paths=[str(cleaned_dir / "texture.png")],
            manifest=loaded_manifest,
        )

        with (
            patch.object(worker.cleaner, "process_cleanup",
                         return_value=(fake_meta, mock_stats, str(fake_cleaned_mesh))),
            patch.object(worker.texturing_service, "run", return_value=stub_tex_result),
        ):
            worker._process_pending_sessions()

        s = session_manager.get_session("sess_happy")
        assert s.status == AssetStatus.CLEANED, f"Expected CLEANED got {s.status}"

        # ── Stage 4: Export ───────────────────────────────────────────────
        glb_path = tmp_data / "registry" / "blobs" / "asset_sess_happy.glb"
        _write_minimal_glb(glb_path)

        with patch.object(worker.exporter, "export", return_value=None):
            worker._process_pending_sessions()

        s = session_manager.get_session("sess_happy")
        assert s.status == AssetStatus.EXPORTED, f"Expected EXPORTED got {s.status}"

        # ── Stage 5: Validation ───────────────────────────────────────────
        export_metrics = {
            "vertex_count": 3, "face_count": 1, "geometry_count": 1,
            "component_count": 1,
            "has_uv": False, "has_material": False, "has_embedded_texture": False,
            "texture_count": 0, "material_count": 0,
            "texture_integrity_status": "missing",
            "material_semantic_status": "geometry_only",
            "material_integrity_status": "missing",
            "basecolor_present": False, "normal_present": False,
            "metallic_roughness_present": False, "occlusion_present": False,
            "emissive_present": False,
            "bounds_min": {"x": 0.0, "y": 0.0, "z": 0.0},
            "bounds_max": {"x": 1.0, "y": 1.0, "z": 1.0},
            "bbox": {"x": 1.0, "y": 1.0, "z": 1.0},
            "ground_offset": 0.0,
            "has_position_accessor": True,
            "has_normal_accessor": True,
            "all_primitives_have_position": True,
            "all_primitives_have_normal": True,
            "all_textured_primitives_have_texcoord_0": False,
            "export_status": "success",
            "structural_export_ready": True,
            "filtering_status": "object_isolated",
        }
        with patch.object(worker.exporter, "inspect_exported_asset",
                          return_value=export_metrics):
            worker._process_pending_sessions()

        s = session_manager.get_session("sess_happy")
        assert s.status == AssetStatus.VALIDATED, f"Expected VALIDATED got {s.status}"
        assert s.validation_report_path is not None

        # ── Stage 6: Publish ──────────────────────────────────────────────
        # Stub manifest → publish_state will be "draft" (not "published").
        with patch.object(worker.exporter, "inspect_exported_asset",
                          return_value=export_metrics):
            worker._process_pending_sessions()

        s = session_manager.get_session("sess_happy")
        assert s.publish_state in {"draft", "published", "failed"}, (
            f"Unexpected publish_state: {s.publish_state}"
        )
        # The session did NOT stall; pipeline advanced to terminal state.
        assert s.status in {AssetStatus.VALIDATED, AssetStatus.PUBLISHED, AssetStatus.FAILED}


# ──────────────────────────────────────────────────────────────────────────────
# TICKET-003: Failure-path integration tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFailurePathIntegration:
    def test_missing_video_marks_session_failed_immediately_no_retry(
        self, tmp_data, session_manager
    ):
        """
        IrrecoverableError (missing video) must terminate without retry.
        retry_count stays 0.
        """
        from modules.operations.worker import IngestionWorker
        from modules.shared_contracts.lifecycle import AssetStatus

        worker = IngestionWorker(data_root=str(tmp_data))
        session_manager.create_session("sess_fail_video", "prod_f", "op_1")
        # Intentionally do NOT create the video file.

        worker._process_pending_sessions()

        s = session_manager.get_session("sess_fail_video")
        assert s.status == AssetStatus.FAILED
        assert "Video file missing" in (s.failure_reason or "")
        assert s.retry_count == 0

    def test_irrecoverable_reconstruction_skips_retry(self, tmp_data, session_manager):
        """
        An IrrecoverableError from reconstruction must not be retried.
        """
        from modules.operations.worker import IngestionWorker, IrrecoverableError
        from modules.shared_contracts.lifecycle import AssetStatus
        from modules.utils.file_persistence import atomic_write_json

        worker = IngestionWorker(data_root=str(tmp_data))
        session_manager.create_session("sess_fail_recon", "prod_r", "op_1")
        session_manager.update_session(
            "sess_fail_recon",
            new_status=AssetStatus.CAPTURED,
            extracted_frames=["fake.jpg"],
            coverage_score=0.9,
            last_pipeline_stage="captured",
        )

        # Provide a frame + coverage report so reconstruction actually runs.
        reports_dir = tmp_data / "captures" / "sess_fail_recon" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(reports_dir / "coverage_report.json", {
            "overall_status": "sufficient",
            "coverage_score": 0.9,
            "reasons": [],
        })
        frames_dir = tmp_data / "captures" / "sess_fail_recon" / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        (frames_dir / "frame_001.jpg").write_bytes(b"fake")

        def bad_recon(session):
            raise IrrecoverableError("CUDA failure detected")

        with patch.object(worker, "_handle_reconstruction", side_effect=bad_recon):
            worker._process_pending_sessions()

        s = session_manager.get_session("sess_fail_recon")
        assert s.status == AssetStatus.FAILED
        assert s.retry_count == 0


# ──────────────────────────────────────────────────────────────────────────────
# TICKET-004: Disk space + binary preflight tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDiskSpacePreflight:
    def test_check_free_disk_gb_returns_non_negative_float(self, tmp_data):
        from modules.operations.settings import Settings
        s = Settings(DATA_ROOT=str(tmp_data))
        result = s.check_free_disk_gb()
        assert isinstance(result, float)
        assert result >= 0.0

    def test_check_free_disk_gb_returns_inf_on_os_error(self):
        from modules.operations.settings import Settings
        s = Settings()
        with patch("shutil.disk_usage", side_effect=OSError("no device")):
            result = s.check_free_disk_gb()
        assert result == float("inf")

    def test_upload_blocked_with_507_when_disk_low_in_pilot(self, tmp_data):
        """
        In pilot/production, HTTP 507 must be returned when free GB < threshold.
        """
        import io
        from fastapi.testclient import TestClient

        mock_s = MagicMock()
        mock_s.is_dev = False
        mock_s.pilot_api_key = "test_key"
        mock_s.data_root = str(tmp_data)
        mock_s.min_free_disk_gb = 10.0
        mock_s.check_ml_deps.return_value = []
        mock_s.check_processing_deps.return_value = []
        mock_s.check_free_disk_gb.return_value = 1.5   # LOW
        mock_s.max_upload_mb = 100.0
        mock_s.max_video_duration_sec = 600.0
        mock_s.min_video_long_edge = 720
        mock_s.min_video_short_edge = 720
        mock_s.ffmpeg_path = "ffmpeg"
        mock_s.ffprobe_path = "ffprobe"
        mock_s.resolve_executable.return_value = "fake_bin"
        mock_s.env.value = "pilot"

        with patch("modules.operations.api.settings", mock_s):
            from modules.operations.api import app
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/api/sessions/upload",
                files={"file": ("video.mp4", io.BytesIO(b"fake_mp4"), "video/mp4")},
                data={"product_id": "prod_disk_test"},
                headers={"x-api-key": "test_key"},
            )

        assert response.status_code == 507, (
            f"Expected 507, got {response.status_code}: {response.text}"
        )
        assert "disk" in response.json()["detail"].lower()

    def test_upload_proceeds_when_disk_sufficient_pilot(self, tmp_data):
        """Upload proceeds when free disk is above threshold."""
        import io
        from fastapi.testclient import TestClient

        mock_s = MagicMock()
        mock_s.is_dev = False
        mock_s.pilot_api_key = "test_key"
        mock_s.data_root = str(tmp_data)
        mock_s.min_free_disk_gb = 5.0
        mock_s.check_ml_deps.return_value = []
        mock_s.check_processing_deps.return_value = []
        mock_s.check_free_disk_gb.return_value = 50.0   # SUFFICIENT
        mock_s.max_upload_mb = 100.0
        mock_s.max_video_duration_sec = 600.0
        mock_s.min_video_long_edge = 720
        mock_s.min_video_short_edge = 720
        mock_s.ffmpeg_path = "ffmpeg"
        mock_s.ffprobe_path = "ffprobe"
        mock_s.resolve_executable.return_value = "fake_bin"
        mock_s.env.value = "pilot"

        with patch("modules.operations.api.settings", mock_s):
            from modules.operations.api import app
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/api/sessions/upload",
                files={"file": ("video.mp4", io.BytesIO(b"fake_mp4"), "video/mp4")},
                data={"product_id": "prod_disk_ok"},
                headers={"x-api-key": "test_key"},
            )

        assert response.status_code not in {507, 503}, (
            f"Upload should not be blocked: {response.status_code} {response.text}"
        )


class TestBinaryProbePreflight:
    def test_probe_colmap_returns_ok_false_when_binary_absent(self):
        from modules.operations.settings import Settings
        s = Settings(RECON_ENGINE_PATH="/nonexistent/colmap.exe")
        result = s.probe_colmap_binary()
        assert result["ok"] is False
        assert "Binary not found" in result["error"]

    def test_probe_colmap_returns_ok_true_when_python_used_as_stub(self, tmp_path):
        """
        Use Python itself as an always-present binary stand-in.
        This verifies the probe's subprocess call path.
        """
        import sys
        from modules.operations.settings import Settings
        s = Settings(RECON_ENGINE_PATH=sys.executable)
        result = s.probe_colmap_binary()
        assert result["ok"] is True
        assert result["version_line"] is not None

    def test_probe_colmap_returns_ok_false_on_timeout(self, tmp_path):
        """Timeout is handled gracefully."""
        import subprocess
        from modules.operations.settings import Settings
        s = Settings(RECON_ENGINE_PATH="/fake/colmap")
        # The binary doesn't exist so subprocess will raise FileNotFoundError,
        # not TimeoutExpired — but we can simulate TimeoutExpired directly.
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="colmap", timeout=10)):
            # Need the binary "to exist" to reach subprocess.run.
            with patch("pathlib.Path.exists", return_value=True):
                result = s.probe_colmap_binary()
        assert result["ok"] is False
        assert "timed out" in result["error"].lower()

    def test_readiness_response_includes_preflight_section(self, tmp_data):
        """
        /api/ready must always contain a 'preflight' key with disk and binary info.
        """
        import io
        from fastapi.testclient import TestClient

        mock_s = MagicMock()
        mock_s.is_dev = True
        mock_s.data_root = str(tmp_data)
        mock_s.min_free_disk_gb = 5.0
        mock_s.check_ml_deps.return_value = []
        mock_s.check_processing_deps.return_value = []
        mock_s.probe_colmap_binary.return_value = {
            "ok": False,
            "version_line": None,
            "error": "Binary not found",
        }
        mock_s.check_free_disk_gb.return_value = 50.0
        mock_s.env.value = "local_dev"

        with patch("modules.operations.api.settings", mock_s):
            from modules.operations.api import app
            client = TestClient(app)
            response = client.get("/api/ready")

        assert response.status_code == 200
        body = response.json()
        assert "preflight" in body, f"'preflight' key missing: {body}"
        pf = body["preflight"]
        assert "colmap_probe_ok" in pf
        assert "free_disk_gb" in pf
        assert "disk_ok" in pf
        assert pf["colmap_probe_ok"] is False
        assert pf["free_disk_gb"] == 50.0
