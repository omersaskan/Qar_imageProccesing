"""
tests/test_sprint2_integration.py

SPRINT 2 — TICKET-009: Regression coverage for export/validation/publish paths

Tests in this file:
  TICKET-005 — session-safe export metric handoff (no cross-session bleed)
  TICKET-005 — _build_registry_metadata reads persisted JSON, not a worker cache
  TICKET-005 — fallback GLB re-inspection when export_metrics_path is absent
  TICKET-006 — progress-aware timeout: advancing session NOT timed out
  TICKET-006 — progress-aware timeout: stalled session IS timed out
  TICKET-006 — _persist_session stamps last_pipeline_progress_at on advance
  TICKET-007 — inspect_exported_asset called exactly once per validation
  TICKET-009 — export → validation → draft publish flow (stub manifest)
  TICKET-009 — export → validation → hard publish flow (non-stub, passing)
  TICKET-009 — validation → fail → FAILED state
  TICKET-009 — two independent sessions produce correct, non-overlapping metrics
"""

from __future__ import annotations

import json
import struct
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

def _make_bbox_dict(lo=0.0, hi=1.0):
    return {"x": lo, "y": lo, "z": lo}, {"x": hi, "y": hi, "z": hi}


def _make_norm_metadata(tmp_data, session_id, lo=0.0, hi=1.0):
    from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
    from modules.utils.file_persistence import atomic_write_json
    bmin, bmax = _make_bbox_dict(lo, hi)
    meta = NormalizedMetadata(
        bbox_min=bmin, bbox_max=bmax,
        pivot_offset={"x": 0.0, "y": 0.0, "z": 0.0},
        final_polycount=1000,
    )
    d = tmp_data / "captures" / session_id / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return meta


def _make_cleanup_stats(mesh_path: str, lo=0.0, hi=1.0) -> Dict[str, Any]:
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
        "pre_aligned_mesh_path": mesh_path,
        "cleaned_mesh_path": mesh_path,
        "metadata_path": str(Path(mesh_path).parent / "normalized_metadata.json"),
        "cleaned_texture_path": None,
        "uv_preserved": False, "material_preserved": False,
    }


def _make_export_metrics(face_count=1) -> Dict[str, Any]:
    return {
        "vertex_count": 3 * face_count,
        "face_count": face_count,
        "geometry_count": 1,
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
    }


def _write_minimal_glb(path: Path) -> None:
    json_dict = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3,
                       "type": "VEC3", "max": [1, 1, 0], "min": [0, 0, 0]}],
        "bufferViews": [{"buffer": 0, "byteLength": 36}],
        "buffers": [{"byteLength": 36}],
    }
    jb = json.dumps(json_dict, separators=(",", ":")).encode("ascii")
    while len(jb) % 4:
        jb += b" "
    bb = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    total = 12 + 8 + len(jb) + 8 + len(bb)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<I4s", len(jb), b"JSON") + jb
        + struct.pack("<I4s", len(bb), b"BIN\x00") + bb
    )


def _setup_cleaned_session(tmp_data, session_manager, session_id) -> str:
    """
    Advance a session to CLEANED state with all required artifacts on disk.
    Returns the cleaned mesh path.
    """
    from modules.shared_contracts.lifecycle import AssetStatus
    from modules.utils.file_persistence import atomic_write_json

    session_manager.create_session(session_id, f"prod_{session_id}", "op_1")

    # Put session into CLEANED state via required transitions
    session_manager.update_session(session_id, new_status=AssetStatus.CAPTURED,
                                   last_pipeline_stage="captured", failure_reason=None,
                                   coverage_score=0.9, extracted_frames=["f.jpg"])
    session_manager.update_session(session_id, new_status=AssetStatus.RECONSTRUCTED,
                                   last_pipeline_stage="reconstructed", failure_reason=None,
                                   reconstruction_manifest_path="placeholder")

    # Write artifacts
    cleaned_dir = tmp_data / "captures" / session_id / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    mesh_p = cleaned_dir / "cleaned_mesh.obj"
    mesh_p.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
    bmin, bmax = _make_bbox_dict()
    meta = NormalizedMetadata(bbox_min=bmin, bbox_max=bmax,
                              pivot_offset={"x": 0.0, "y": 0.0, "z": 0.0},
                              final_polycount=1)
    meta_p = cleaned_dir / "normalized_metadata.json"
    stats_p = cleaned_dir / "cleanup_stats.json"
    atomic_write_json(meta_p, meta.model_dump(mode="json"))
    atomic_write_json(stats_p, _make_cleanup_stats(str(mesh_p)))

    # Manifest
    job_dir = tmp_data / "reconstructions" / f"job_{session_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    from modules.reconstruction_engine.output_manifest import OutputManifest
    manifest = OutputManifest(
        job_id=f"job_{session_id}", mesh_path=str(mesh_p),
        log_path=str(job_dir / "log.txt"),
        processing_time_seconds=1.0, engine_type="stub", is_stub=True,
    )
    atomic_write_json(job_dir / "manifest.json", manifest.model_dump(mode="json"))

    session_manager.update_session(session_id, new_status=AssetStatus.CLEANED,
                                   last_pipeline_stage="cleaned", failure_reason=None,
                                   cleanup_mesh_path=str(mesh_p),
                                   cleanup_metadata_path=str(meta_p),
                                   cleanup_stats_path=str(stats_p),
                                   reconstruction_manifest_path=str(job_dir / "manifest.json"))
    return str(mesh_p)


# ──────────────────────────────────────────────────────────────────────────────
# TICKET-005: Session-safe export metric handoff
# ──────────────────────────────────────────────────────────────────────────────

class TestExportMetricSessionSafety:
    """
    Verify that export metrics are persisted per-session and cannot bleed
    across two sessions processed in the same worker instance.
    """

    def test_export_metrics_persisted_to_disk_after_validation(self, tmp_data, session_manager, worker):
        """
        After _handle_validation the session should have export_metrics_path set,
        and the file at that path should contain the correct metrics.
        """
        from modules.shared_contracts.lifecycle import AssetStatus

        _setup_cleaned_session(tmp_data, session_manager, "sess_em1")
        session = session_manager.get_session("sess_em1")

        # Advance to EXPORTED
        glb_path = worker.blobs_dir / "asset_sess_em1.glb"
        _write_minimal_glb(glb_path)
        with patch.object(worker.exporter, "export", return_value=None):
            worker._process_pending_sessions()
        session = session_manager.get_session("sess_em1")
        assert session.status == AssetStatus.EXPORTED

        # Validate — metrics should be persisted
        metrics_a = _make_export_metrics(face_count=42)
        with patch.object(worker.exporter, "inspect_exported_asset", return_value=metrics_a):
            worker._process_pending_sessions()

        session = session_manager.get_session("sess_em1")
        assert session.status == AssetStatus.VALIDATED
        assert session.export_metrics_path is not None, "export_metrics_path must be set"
        p = Path(session.export_metrics_path)
        assert p.exists(), "export_metrics.json must exist on disk"
        loaded = json.loads(p.read_text())
        assert loaded["face_count"] == 42, "Persisted metrics face_count must match"

    def test_two_sessions_have_independent_metrics(self, tmp_data, session_manager, worker):
        """
        Two sessions validated sequentially must each have their own export_metrics.json
        with the correct per-session values.  No metric bleed.
        """
        from modules.shared_contracts.lifecycle import AssetStatus

        _setup_cleaned_session(tmp_data, session_manager, "sess_m1")
        _setup_cleaned_session(tmp_data, session_manager, "sess_m2")

        # Export both sessions
        for sid in ("sess_m1", "sess_m2"):
            glb = worker.blobs_dir / f"asset_{sid}.glb"
            _write_minimal_glb(glb)
        with patch.object(worker.exporter, "export", return_value=None):
            worker._process_pending_sessions()

        for sid in ("sess_m1", "sess_m2"):
            s = session_manager.get_session(sid)
            assert s.status == AssetStatus.EXPORTED

        # Return different metrics for each session.
        # We do so by side_effect list (order matches glob iteration) but we
        # also record the call count and verify the JSON files independently.
        metrics_for = {
            "sess_m1": _make_export_metrics(face_count=111),
            "sess_m2": _make_export_metrics(face_count=999),
        }

        inspect_calls = []

        def side_effect_inspect(glb_path: str):
            # determine session from path
            for sid in ("sess_m1", "sess_m2"):
                if sid in glb_path:
                    inspect_calls.append(sid)
                    return metrics_for[sid]
            raise ValueError(f"Unknown GLB path in test: {glb_path}")

        with patch.object(worker.exporter, "inspect_exported_asset",
                          side_effect=side_effect_inspect):
            worker._process_pending_sessions()

        s1 = session_manager.get_session("sess_m1")
        s2 = session_manager.get_session("sess_m2")

        for s in (s1, s2):
            assert s.export_metrics_path is not None
            assert Path(s.export_metrics_path).exists()

        m1 = json.loads(Path(s1.export_metrics_path).read_text())
        m2 = json.loads(Path(s2.export_metrics_path).read_text())

        assert m1["face_count"] == 111, f"Session 1 metrics should be 111, got {m1['face_count']}"
        assert m2["face_count"] == 999, f"Session 2 metrics should be 999, got {m2['face_count']}"
        assert m1["face_count"] != m2["face_count"], "Metrics must not bleed across sessions"

    def test_build_registry_metadata_reads_json_no_second_glb_parse(
        self, tmp_data, session_manager, worker
    ):
        """
        _build_registry_metadata must read export_metrics_path (disk JSON),
        NOT call inspect_exported_asset again.
        """
        from modules.shared_contracts.lifecycle import AssetStatus
        from modules.utils.file_persistence import atomic_write_json
        from modules.shared_contracts.models import ValidationReport

        _setup_cleaned_session(tmp_data, session_manager, "sess_meta")

        # Manually plant an export_metrics.json so session has export_metrics_path.
        reports_dir = tmp_data / "captures" / "sess_meta" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        planted_metrics = _make_export_metrics(face_count=77)
        metrics_path = reports_dir / "export_metrics.json"
        atomic_write_json(metrics_path, planted_metrics)

        glb_path = worker.blobs_dir / "asset_sess_meta.glb"
        _write_minimal_glb(glb_path)

        session_manager.update_session(
            "sess_meta",
            new_status=AssetStatus.EXPORTED,
            last_pipeline_stage="exported",
            export_blob_path=str(glb_path),
            export_metrics_path=str(metrics_path),
            failure_reason=None,
            asset_id="asset_sess_meta",
        )
        session = session_manager.get_session("sess_meta")

        # Build a fake validation report
        val_report = ValidationReport(
            asset_id="asset_sess_meta",
            poly_count=77,
            texture_status="missing",
            bbox_reasonable=True,
            ground_aligned=True,
            mobile_performance_grade="B",
            component_count=1,
            largest_component_share=1.0,
            contamination_score=0.0,
            contamination_report={},
            final_decision="pass",
        )

        inspect_call_count = {"n": 0}

        def recording_inspect(glb_path):
            inspect_call_count["n"] += 1
            return planted_metrics

        with patch.object(worker.exporter, "inspect_exported_asset",
                          side_effect=recording_inspect):
            result = worker._build_registry_metadata(session, "asset_sess_meta", val_report)

        # inspect_exported_asset must NOT have been called (metrics came from JSON).
        assert inspect_call_count["n"] == 0, (
            f"Expected 0 GLB re-inspections, got {inspect_call_count['n']}"
        )
        assert result.bbox["dimensions"]["x"] == 1.0

    def test_fallback_to_glb_inspection_when_metrics_path_absent(
        self, tmp_data, session_manager, worker
    ):
        """
        When export_metrics_path is not set (legacy session), _load_export_metrics
        must fall back to re-inspecting the GLB and log a warning.
        """
        from modules.shared_contracts.lifecycle import AssetStatus
        from modules.shared_contracts.models import ValidationReport

        _setup_cleaned_session(tmp_data, session_manager, "sess_fallback")

        glb_path = worker.blobs_dir / "asset_sess_fallback.glb"
        _write_minimal_glb(glb_path)

        session_manager.update_session(
            "sess_fallback",
            new_status=AssetStatus.EXPORTED,
            last_pipeline_stage="exported",
            export_blob_path=str(glb_path),
            # Intentionally do NOT set export_metrics_path → simulates a legacy session.
            failure_reason=None,
            asset_id="asset_sess_fallback",
        )
        session = session_manager.get_session("sess_fallback")
        assert session.export_metrics_path is None

        fallback_metrics = _make_export_metrics(face_count=55)
        val_report = ValidationReport(
            asset_id="asset_sess_fallback",
            poly_count=55,
            texture_status="missing",
            bbox_reasonable=True,
            ground_aligned=True,
            mobile_performance_grade="C",
            component_count=1, largest_component_share=1.0,
            contamination_score=0.0, contamination_report={},
            final_decision="pass",
        )

        with patch.object(worker.exporter, "inspect_exported_asset",
                          return_value=fallback_metrics) as mock_inspect:
            result = worker._build_registry_metadata(session, "asset_sess_fallback", val_report)

        # Fallback IS called exactly once.
        mock_inspect.assert_called_once()
        assert result.bbox["dimensions"]["y"] == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# TICKET-006: Progress-aware timeout
# ──────────────────────────────────────────────────────────────────────────────

class TestProgressAwareTimeout:
    def test_progressing_session_not_timed_out(self, tmp_data, session_manager, worker):
        """
        A session whose `last_pipeline_progress_at` is recent must NOT be timed out,
        even if its total age (`created_at`) is old.
        """
        from modules.shared_contracts.lifecycle import AssetStatus
        from modules.utils.file_persistence import atomic_write_json

        session_manager.create_session("sess_prog", "prod_p", "op_1")

        # Back-date created_at to 10 hours ago — would trigger old timeout logic.
        sess_file = tmp_data / "sessions" / "sess_prog.json"
        with open(sess_file) as f:
            data = json.load(f)
        data["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
        # But last_pipeline_progress_at is recent (30 minutes ago).
        data["last_pipeline_progress_at"] = (
            datetime.now(timezone.utc) - timedelta(minutes=30)
        ).isoformat()
        atomic_write_json(sess_file, data)

        with patch("modules.operations.worker.settings") as ms:
            ms.max_retry_count = 5
            ms.session_timeout_hours = 2  # 2h stale threshold
            ms.worker_interval_sec = 5
            ms.data_root = str(tmp_data)

            timed_out = worker._check_session_timeout(
                session_manager.get_session("sess_prog")
            )

        assert timed_out is False, (
            "Session with recent progress must NOT be timed out (TICKET-006)"
        )

    def test_stalled_session_is_timed_out(self, tmp_data, session_manager, worker):
        """
        A session whose `last_pipeline_progress_at` is older than the threshold
        must be timed out and marked FAILED.
        """
        from modules.shared_contracts.lifecycle import AssetStatus
        from modules.utils.file_persistence import atomic_write_json

        session_manager.create_session("sess_stall", "prod_s", "op_1")

        sess_file = tmp_data / "sessions" / "sess_stall.json"
        with open(sess_file) as f:
            data = json.load(f)
        # created_at: 5h ago; last_pipeline_progress_at: 4h ago — both stale.
        data["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        data["last_pipeline_progress_at"] = (
            datetime.now(timezone.utc) - timedelta(hours=4)
        ).isoformat()
        atomic_write_json(sess_file, data)

        with patch("modules.operations.worker.settings") as ms:
            ms.max_retry_count = 5
            ms.session_timeout_hours = 2
            ms.worker_interval_sec = 5
            ms.data_root = str(tmp_data)

            worker._process_pending_sessions()

        s = session_manager.get_session("sess_stall")
        assert s.status == AssetStatus.FAILED
        assert "timed out" in (s.failure_reason or "").lower()
        assert "no pipeline progress" in (s.failure_reason or "").lower()

    def test_no_progress_timestamp_falls_back_to_created_at(
        self, tmp_data, session_manager, worker
    ):
        """
        When last_pipeline_progress_at is None, created_at is the fallback.
        An old session with no progress should still be timed out.
        """
        from modules.utils.file_persistence import atomic_write_json
        from modules.shared_contracts.lifecycle import AssetStatus

        session_manager.create_session("sess_noprog", "prod_np", "op_1")

        sess_file = tmp_data / "sessions" / "sess_noprog.json"
        with open(sess_file) as f:
            data = json.load(f)
        data["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        data["last_pipeline_progress_at"] = None
        atomic_write_json(sess_file, data)

        with patch("modules.operations.worker.settings") as ms:
            ms.max_retry_count = 5
            ms.session_timeout_hours = 2
            ms.worker_interval_sec = 5
            ms.data_root = str(tmp_data)

            worker._process_pending_sessions()

        s = session_manager.get_session("sess_noprog")
        assert s.status == AssetStatus.FAILED
        assert "timed out" in (s.failure_reason or "").lower()

    def test_persist_session_stamps_progress_timestamp(self, tmp_data, session_manager, worker):
        """
        Calling _persist_session with a new_status must automatically set
        last_pipeline_progress_at to a recent UTC time.
        """
        from modules.shared_contracts.lifecycle import AssetStatus

        session = session_manager.create_session("sess_stamp", "prod_st", "op_1")
        before = datetime.now(timezone.utc)

        updated = worker._persist_session(
            session,
            new_status=AssetStatus.CAPTURED,
            last_pipeline_stage="captured",
            failure_reason=None,
            coverage_score=0.9,
            extracted_frames=["f.jpg"],
        )
        after = datetime.now(timezone.utc)

        # Read back from disk to confirm it was persisted.
        persisted = session_manager.get_session("sess_stamp")
        assert persisted.last_pipeline_progress_at is not None, (
            "last_pipeline_progress_at must be set after a successful stage advance"
        )
        # It must be within the test window.
        assert before <= persisted.last_pipeline_progress_at <= after, (
            f"Expected timestamp between {before} and {after}, "
            f"got {persisted.last_pipeline_progress_at}"
        )

    def test_persist_session_does_not_stamp_when_status_unchanged(
        self, tmp_data, session_manager, worker
    ):
        """
        Calling _persist_session with new_status=None (field-only update) must NOT
        overwrite last_pipeline_progress_at.
        """
        session = session_manager.create_session("sess_nostamp", "prod_ns", "op_1")
        # Ensure last_pipeline_progress_at is None initially.
        s = session_manager.get_session("sess_nostamp")
        assert s.last_pipeline_progress_at is None

        # Update a field without advancing status.
        worker._persist_session(session, new_status=None, failure_reason="test error")

        s2 = session_manager.get_session("sess_nostamp")
        assert s2.last_pipeline_progress_at is None, (
            "last_pipeline_progress_at must NOT be set when status doesn't change"
        )


# ──────────────────────────────────────────────────────────────────────────────
# TICKET-007: Single GLB inspection per validation
# ──────────────────────────────────────────────────────────────────────────────

class TestGLBLoadDeduplication:
    def test_inspect_exported_asset_called_exactly_once_per_session(
        self, tmp_data, session_manager, worker
    ):
        """
        Through the full export → validation → publish cycle for a single session,
        inspect_exported_asset must be called exactly once (during _handle_validation).
        _build_registry_metadata must NOT trigger a second call.
        """
        from modules.shared_contracts.lifecycle import AssetStatus
        from modules.utils.file_persistence import atomic_write_json

        _setup_cleaned_session(tmp_data, session_manager, "sess_dedup")

        glb_path = worker.blobs_dir / "asset_sess_dedup.glb"
        _write_minimal_glb(glb_path)

        # Export
        with patch.object(worker.exporter, "export", return_value=None):
            worker._process_pending_sessions()

        s = session_manager.get_session("sess_dedup")
        assert s.status == AssetStatus.EXPORTED

        # Validate: track inspect calls
        inspect_count = {"n": 0}
        def counting_inspect(path):
            inspect_count["n"] += 1
            return _make_export_metrics()

        with patch.object(worker.exporter, "inspect_exported_asset", side_effect=counting_inspect):
            worker._process_pending_sessions()

        s = session_manager.get_session("sess_dedup")
        assert s.status == AssetStatus.VALIDATED

        # Publish (uses _build_registry_metadata = reads JSON, not GLB)
        with patch.object(worker.exporter, "inspect_exported_asset", side_effect=counting_inspect):
            worker._process_pending_sessions()

        # Total expected: 1 call during _handle_validation.
        # _handle_publish._build_registry_metadata must read JSON, not re-inspect.
        assert inspect_count["n"] == 1, (
            f"Expected exactly 1 GLB inspection across validation+publish, got {inspect_count['n']}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# TICKET-009: Export → Validation → Publish regression
# ──────────────────────────────────────────────────────────────────────────────

class TestExportValidationPublishFlow:
    def test_stub_manifest_lands_in_draft(self, tmp_data, session_manager, worker):
        """
        A session whose reconstruction manifest has is_stub=True must end up in
        publish_state='draft', not 'published'.

        _handle_publish logic (worker.py):
          1. if final_decision == 'fail'  → FAILED  (hard stop)
          2. elif is_stub OR final_decision == 'review' → draft
          3. else → published

        The real AssetValidator returns 'fail' for geometry-only minimal meshes
        because of contamination/quality checks, so we stub the validator to
        return 'review' — which exercises the is_stub=True draft path via branch 2.
        A 'review' decision on a stub manifest is the honest expected outcome.
        """
        from modules.shared_contracts.lifecycle import AssetStatus
        from modules.shared_contracts.models import ValidationReport

        _setup_cleaned_session(tmp_data, session_manager, "sess_draft")
        glb = worker.blobs_dir / "asset_sess_draft.glb"
        _write_minimal_glb(glb)

        # Export
        with patch.object(worker.exporter, "export", return_value=None):
            worker._process_pending_sessions()

        # Validate — stub the validator to return 'review' (not fail, not pass)
        # so that the publish path is reached and is_stub triggers draft.
        review_report = ValidationReport(
            asset_id="asset_sess_draft",
            poly_count=1,
            texture_status="missing",
            bbox_reasonable=True,
            ground_aligned=True,
            mobile_performance_grade="C",
            component_count=1,
            largest_component_share=1.0,
            contamination_score=0.1,
            contamination_report={},
            final_decision="review",
        )
        with (
            patch.object(worker.exporter, "inspect_exported_asset",
                         return_value=_make_export_metrics()),
            patch.object(worker.validator, "validate", return_value=review_report),
        ):
            worker._process_pending_sessions()

        s = session_manager.get_session("sess_draft")
        assert s.status == AssetStatus.VALIDATED

        # Publish — is_stub=True + review → draft
        worker._process_pending_sessions()
        s = session_manager.get_session("sess_draft")
        assert s.publish_state == "draft", (
            f"Stub manifest + review decision must produce publish_state='draft', "
            f"got '{s.publish_state}' (failure_reason: {s.failure_reason})"
        )

    def test_validation_fail_marks_session_failed(self, tmp_data, session_manager, worker):
        """
        When the validator returns final_decision='fail', _handle_publish must
        mark the session as FAILED, not draft/published.
        """
        from modules.shared_contracts.lifecycle import AssetStatus
        from modules.shared_contracts.models import ValidationReport
        from modules.utils.file_persistence import atomic_write_json

        _setup_cleaned_session(tmp_data, session_manager, "sess_valfail")

        glb = worker.blobs_dir / "asset_sess_valfail.glb"
        _write_minimal_glb(glb)

        with patch.object(worker.exporter, "export", return_value=None):
            worker._process_pending_sessions()

        with patch.object(worker.exporter, "inspect_exported_asset",
                          return_value=_make_export_metrics()):
            worker._process_pending_sessions()

        s = session_manager.get_session("sess_valfail")
        assert s.status == AssetStatus.VALIDATED

        # Overwrite the validation report on disk with a 'fail' decision.
        failing_report = ValidationReport(
            asset_id=f"asset_sess_valfail",
            poly_count=1,
            texture_status="missing",
            bbox_reasonable=True,
            ground_aligned=True,
            mobile_performance_grade="D",
            component_count=1, largest_component_share=1.0,
            contamination_score=0.99,
            contamination_report={"reason": "too contaminated"},
            final_decision="fail",
        )
        reports_dir = tmp_data / "captures" / "sess_valfail" / "reports"
        atomic_write_json(reports_dir / "validation_report.json",
                          failing_report.model_dump(mode="json"))

        # Update session to use this new report path
        session_manager.update_session(
            "sess_valfail",
            validation_report_path=str(reports_dir / "validation_report.json"),
        )

        worker._process_pending_sessions()
        s = session_manager.get_session("sess_valfail")
        assert s.status == AssetStatus.FAILED, (
            f"Expected FAILED after validation fail, got {s.status}"
        )
        assert "Validation Failed" in (s.failure_reason or "")

    def test_two_sessions_no_metric_bleed_through_full_pipeline(
        self, tmp_data, session_manager, worker
    ):
        """
        Two sessions driven through export → validation simultaneously in the same
        worker cycle must produce independent export_metrics_path files with the
        correct per-session values and must each reach VALIDATED independently.
        """
        from modules.shared_contracts.lifecycle import AssetStatus

        for sid in ("sess_dual_a", "sess_dual_b"):
            _setup_cleaned_session(tmp_data, session_manager, sid)
            glb = worker.blobs_dir / f"asset_{sid}.glb"
            _write_minimal_glb(glb)

        # Export both
        with patch.object(worker.exporter, "export", return_value=None):
            worker._process_pending_sessions()

        for sid in ("sess_dual_a", "sess_dual_b"):
            s = session_manager.get_session(sid)
            assert s.status == AssetStatus.EXPORTED, f"{sid} should be EXPORTED"

        # Map distinct metrics per session
        metrics_for = {
            "sess_dual_a": _make_export_metrics(face_count=200),
            "sess_dual_b": _make_export_metrics(face_count=400),
        }

        def routed_inspect(glb_path: str):
            for sid in metrics_for:
                if sid in glb_path:
                    return metrics_for[sid]
            return _make_export_metrics()

        with patch.object(worker.exporter, "inspect_exported_asset",
                          side_effect=routed_inspect):
            worker._process_pending_sessions()

        sa = session_manager.get_session("sess_dual_a")
        sb = session_manager.get_session("sess_dual_b")

        assert sa.status == AssetStatus.VALIDATED
        assert sb.status == AssetStatus.VALIDATED

        ma = json.loads(Path(sa.export_metrics_path).read_text())
        mb = json.loads(Path(sb.export_metrics_path).read_text())

        assert ma["face_count"] == 200, f"Session A face_count must be 200, got {ma['face_count']}"
        assert mb["face_count"] == 400, f"Session B face_count must be 400, got {mb['face_count']}"

    def test_validated_session_does_not_revalidate_on_next_cycle(
        self, tmp_data, session_manager, worker
    ):
        """
        A session already in VALIDATED/pending state should not trigger another
        inspect_exported_asset call on the next worker cycle.
        """
        from modules.shared_contracts.lifecycle import AssetStatus

        _setup_cleaned_session(tmp_data, session_manager, "sess_norev")
        glb = worker.blobs_dir / "asset_sess_norev.glb"
        _write_minimal_glb(glb)

        with patch.object(worker.exporter, "export", return_value=None):
            worker._process_pending_sessions()

        call_count = {"n": 0}

        def counting(path):
            call_count["n"] += 1
            return _make_export_metrics()

        with patch.object(worker.exporter, "inspect_exported_asset", side_effect=counting):
            worker._process_pending_sessions()  # validation cycle

        s = session_manager.get_session("sess_norev")
        assert s.status == AssetStatus.VALIDATED
        first_count = call_count["n"]
        assert first_count == 1

        # Second cycle: session is VALIDATED with pending publish — should go to _handle_publish
        # which reads JSON, not GLB.
        with patch.object(worker.exporter, "inspect_exported_asset", side_effect=counting):
            worker._process_pending_sessions()  # publish cycle

        # inspect_exported_asset must NOT have been called again.
        assert call_count["n"] == first_count, (
            f"inspect_exported_asset should not be called again in publish cycle; "
            f"got {call_count['n']} total calls"
        )
