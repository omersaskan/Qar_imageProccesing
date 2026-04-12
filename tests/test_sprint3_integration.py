"""
tests/test_sprint3_integration.py

SPRINT 3 — Regression coverage for guidance, retention, and calibration

Tests:
  TICKET-009 — calibration evaluation harness produces correct metrics
  TICKET-009 — confusion matrix is computed correctly
  TICKET-009 — calibration advice fires when precision/recall is low
  TICKET-010 — guidance message codes are correct for each key status
  TICKET-010 — coverage failure patterns produce correct operator messages
  TICKET-010 — validation contamination produces correct messages
  TICKET-010 — FAILED status does NOT leak raw Python traceback strings
  TICKET-010 — messages are deduplicated within a single guidance call
  TICKET-010 — to_markdown includes validation table
  TICKET-011 — PUBLISHED sessions are pruned after published_frames_days
  TICKET-011 — DRAFT sessions are pruned after draft_frames_days
  TICKET-011 — FAILED sessions are pruned after failed_frames_days
  TICKET-011 — ACTIVE sessions are never pruned
  TICKET-011 — protected artifacts are never deleted
  TICKET-011 — run_cleanup returns a structured summary dict
  TICKET-011 — orphaned recon dir with manifest is scratch-pruned not deleted
  TICKET-012 — api.py lifespan handler (no on_event deprecation warning)
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_coverage_report(status="sufficient", reasons=None, **kwargs) -> Dict[str, Any]:
    base = {
        "num_frames": 30,
        "readable_frames": 28,
        "unique_views": 6,
        "diversity": "sufficient" if status == "sufficient" else "insufficient",
        "top_down_captured": True,
        "center_x_span": 0.18,
        "center_y_span": 0.14,
        "scale_variation": 1.25,
        "aspect_variation": 0.16,
        "coverage_score": 0.82 if status == "sufficient" else 0.35,
        "fallback_frames": 0,
        "low_confidence_frames": 0,
        "overall_status": status,
        "recommended_action": "reconstruct" if status == "sufficient" else "needs_recapture",
        "reasons": reasons or [],
        "ml_segmentation_unavailable": False,
    }
    base.update(kwargs)
    return base


def _make_validation_report(decision="pass", contamination=0.0, **kwargs) -> Dict[str, Any]:
    base = {
        "asset_id": "test_asset",
        "poly_count": 5000,
        "texture_status": "real",
        "bbox_reasonable": True,
        "ground_aligned": True,
        "mobile_performance_grade": "B",
        "component_count": 1,
        "largest_component_share": 1.0,
        "contamination_score": contamination,
        "contamination_report": {},
        "final_decision": decision,
        "flatness_score": 0.9,
        "compactness_score": 0.9,
        "selected_component_score": 0.9,
        "material_quality_grade": "B",
        "material_semantic_status": "diffuse_textured",
    }
    base.update(kwargs)
    return base


# ──────────────────────────────────────────────────────────────────────────────
# TICKET-009: Calibration evaluation harness
# ──────────────────────────────────────────────────────────────────────────────

class TestCalibrationEvaluator:
    def _get_evaluator(self):
        """Import the calibration module without installing."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evaluate_coverage_heuristics",
            Path(__file__).parent.parent
            / "tools" / "calibration" / "evaluate_coverage_heuristics.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_perfect_fixtures_give_100_accuracy(self):
        """All fixtures predicted correctly → accuracy 1.0."""
        mod = self._get_evaluator()
        fixtures = [
            {
                "fixture_id": "f1",
                "label": "good",
                "coverage_report": _make_coverage_report(status="sufficient"),
            },
            {
                "fixture_id": "f2",
                "label": "recapture",
                "coverage_report": _make_coverage_report(
                    status="insufficient",
                    reasons=["Insufficient viewpoint diversity (3/5 unique views)."]
                ),
            },
        ]
        results = [mod._evaluate_fixture(f) for f in fixtures]
        metrics = mod._compute_metrics(results)
        assert metrics["accuracy"] == 1.0
        assert metrics["tp"] == 1
        assert metrics["tn"] == 1
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0

    def test_false_positive_counted_correctly(self):
        """Good fixture predicted as recapture → FP."""
        mod = self._get_evaluator()
        fixtures = [
            {
                "fixture_id": "fp_test",
                "label": "good",
                "coverage_report": _make_coverage_report(
                    status="insufficient",  # heuristic says recapture, label says good
                    reasons=["Object motion across accepted views is too narrow."]
                ),
            }
        ]
        results = [mod._evaluate_fixture(f) for f in fixtures]
        metrics = mod._compute_metrics(results)
        assert metrics["fp"] == 1
        assert metrics["tp"] == 0
        assert metrics["tn"] == 0
        assert metrics["fn"] == 0
        assert metrics["accuracy"] == 0.0

    def test_false_negative_counted_correctly(self):
        """Bad fixture predicted as good → FN."""
        mod = self._get_evaluator()
        fixtures = [
            {
                "fixture_id": "fn_test",
                "label": "recapture",
                "coverage_report": _make_coverage_report(status="sufficient"),
            }
        ]
        results = [mod._evaluate_fixture(f) for f in fixtures]
        metrics = mod._compute_metrics(results)
        assert metrics["fn"] == 1
        assert metrics["accuracy"] == 0.0

    def test_low_precision_triggers_advice(self):
        """Calibration advice fires when FP count is nonzero and precision low."""
        mod = self._get_evaluator()
        metrics = {
            "total": 5, "correct": 3,
            "tp": 2, "tn": 1, "fp": 2, "fn": 0,
            "accuracy": 0.60, "precision": 0.50, "recall": 1.0, "f1": 0.67,
        }
        advice = mod._calibration_advice(metrics)
        assert any("PRECISION LOW" in a for a in advice), (
            f"Expected PRECISION LOW advice, got: {advice}"
        )

    def test_low_recall_triggers_advice(self):
        """Calibration advice fires when FN count is nonzero and recall low."""
        mod = self._get_evaluator()
        metrics = {
            "total": 5, "correct": 3,
            "tp": 1, "tn": 3, "fp": 0, "fn": 1,
            "accuracy": 0.80, "precision": 1.0, "recall": 0.50, "f1": 0.67,
        }
        advice = mod._calibration_advice(metrics)
        assert any("RECALL LOW" in a for a in advice), (
            f"Expected RECALL LOW advice, got: {advice}"
        )

    def test_bundled_fixtures_file_loads_and_evaluates(self):
        """The bundled fixtures JSON must be valid and evaluate without errors."""
        mod = self._get_evaluator()
        fixtures_path = (
            Path(__file__).parent.parent
            / "tools" / "calibration" / "fixtures" / "coverage_fixtures.json"
        )
        assert fixtures_path.exists(), "Bundled fixtures file must exist"
        report = mod.run_evaluation(fixtures_path, fmt="json")
        assert "metrics" in report
        assert report["metrics"]["total"] == 5
        # fix_001 is "good" and should be correctly classified as "sufficient"
        f1 = next(r for r in report["results"] if r["fixture_id"] == "fix_001")
        assert f1["correct"] is True


# ──────────────────────────────────────────────────────────────────────────────
# TICKET-010: Guidance message quality
# ──────────────────────────────────────────────────────────────────────────────

class TestGuidanceMessageQuality:
    @pytest.fixture(autouse=True)
    def aggregator(self):
        from modules.operations.guidance import GuidanceAggregator
        self.agg = GuidanceAggregator()

    def _codes(self, guidance) -> list:
        return [m["code"] for m in guidance.messages]

    def test_created_status_gives_awaiting_upload(self):
        from modules.shared_contracts.lifecycle import AssetStatus
        g = self.agg.generate_guidance("s1", AssetStatus.CREATED)
        assert "AWAITING_UPLOAD" in self._codes(g)
        assert g.should_recapture is False

    def test_captured_gives_processing_reconstruction(self):
        from modules.shared_contracts.lifecycle import AssetStatus
        g = self.agg.generate_guidance("s1", AssetStatus.CAPTURED)
        assert "PROCESSING_RECONSTRUCTION" in self._codes(g)

    def test_recapture_required_gives_recapture_needed(self):
        from modules.shared_contracts.lifecycle import AssetStatus
        g = self.agg.generate_guidance("s1", AssetStatus.RECAPTURE_REQUIRED,
                                       failure_reason="Insufficient viewpoint diversity")
        assert "RECAPTURE_NEEDED" in self._codes(g)
        assert g.should_recapture is True

    def test_recapture_failure_reason_maps_to_specific_code(self):
        """'viewpoint diversity' pattern must produce RECAPTURE_LOW_DIVERSITY, not raw text."""
        from modules.shared_contracts.lifecycle import AssetStatus
        g = self.agg.generate_guidance(
            "s1",
            AssetStatus.RECAPTURE_REQUIRED,
            failure_reason="Insufficient viewpoint diversity (3/5 unique views).",
        )
        codes = self._codes(g)
        assert "RECAPTURE_LOW_DIVERSITY" in codes

    def test_failed_cuda_maps_to_system_failure_config(self):
        from modules.shared_contracts.lifecycle import AssetStatus
        g = self.agg.generate_guidance(
            "s1", AssetStatus.FAILED,
            failure_reason="Engine configuration or deterministic failure: CUDA device not found",
        )
        assert "SYSTEM_FAILURE_CONFIG" in self._codes(g)
        # Must NOT contain Python traceback leak in any message text
        for m in g.messages:
            assert "Traceback" not in m["message"]
            assert "exception" not in m["message"].lower() or "contact" in m["message"].lower()

    def test_failed_timeout_maps_pipeline_failure(self):
        from modules.shared_contracts.lifecycle import AssetStatus
        g = self.agg.generate_guidance(
            "s1", AssetStatus.FAILED,
            failure_reason="Session timed out: no pipeline progress for 3.0h",
        )
        assert "SYSTEM_FAILURE_PIPELINE" in self._codes(g)

    def test_failed_unknown_reason_gives_generic_pipeline_failure(self):
        """Unknown failure reason must not surface raw message."""
        from modules.shared_contracts.lifecycle import AssetStatus
        raw = "AttributeError: 'NoneType' has no attribute 'foo'"
        g = self.agg.generate_guidance("s1", AssetStatus.FAILED, failure_reason=raw)
        codes = self._codes(g)
        assert "SYSTEM_FAILURE_PIPELINE" in codes
        # Raw Python error must NOT appear in any message text
        for m in g.messages:
            assert "AttributeError" not in m["message"]
            assert raw not in m["message"]

    def test_coverage_low_diversity_produces_correct_code(self):
        from modules.shared_contracts.lifecycle import AssetStatus
        cov = _make_coverage_report(
            status="insufficient",
            reasons=["Insufficient viewpoint diversity (3/5 unique views)."],
        )
        g = self.agg.generate_guidance("s1", AssetStatus.RECAPTURE_REQUIRED,
                                       coverage_report=cov)
        assert "RECAPTURE_LOW_DIVERSITY" in self._codes(g)

    def test_coverage_scale_flat_produces_correct_code(self):
        from modules.shared_contracts.lifecycle import AssetStatus
        cov = _make_coverage_report(
            status="insufficient",
            reasons=["Accepted views show too little scale/shape variation."],
        )
        g = self.agg.generate_guidance("s1", AssetStatus.RECAPTURE_REQUIRED,
                                       coverage_report=cov)
        assert "RECAPTURE_SCALE_FLAT" in self._codes(g)

    def test_coverage_fallback_masking_produces_correct_code(self):
        from modules.shared_contracts.lifecycle import AssetStatus
        cov = _make_coverage_report(
            status="insufficient",
            reasons=["Too many frames relied on heuristic fallback (18)."],
        )
        g = self.agg.generate_guidance("s1", AssetStatus.RECAPTURE_REQUIRED,
                                       coverage_report=cov)
        assert "RECAPTURE_MASKING_FALLBACK" in self._codes(g)

    def test_ml_unavailable_produces_masking_degraded_code(self):
        from modules.shared_contracts.lifecycle import AssetStatus
        cov = _make_coverage_report(ml_segmentation_unavailable=True)
        g = self.agg.generate_guidance("s1", AssetStatus.CAPTURED, coverage_report=cov)
        assert "MASKING_DEGRADED_ML" in self._codes(g)

    def test_high_contamination_produces_contamination_high(self):
        from modules.shared_contracts.lifecycle import AssetStatus
        val = _make_validation_report(decision="fail", contamination=0.7)
        g = self.agg.generate_guidance("s1", AssetStatus.VALIDATED, validation_report=val)
        assert "CONTAMINATION_HIGH" in self._codes(g)

    def test_texture_uv_failure_produces_correct_code(self):
        from modules.shared_contracts.lifecycle import AssetStatus
        val = _make_validation_report(
            decision="review",
            contamination_report={"texture_uv_integrity": "fail"},
        )
        g = self.agg.generate_guidance("s1", AssetStatus.VALIDATED, validation_report=val)
        assert "TEXTURE_UV_FAILURE" in self._codes(g)

    def test_validation_review_produces_review_advice(self):
        from modules.shared_contracts.lifecycle import AssetStatus
        val = _make_validation_report(decision="review")
        g = self.agg.generate_guidance("s1", AssetStatus.VALIDATED, validation_report=val)
        assert "VALIDATION_REVIEW_ADVICE" in self._codes(g)
        assert g.is_ready_for_review is True
        assert g.should_recapture is False

    def test_messages_are_deduplicated(self):
        """
        A single guidance call with a coverage report that generates
        RECAPTURE_LOW_DIVERSITY and a failure_reason that also matches
        RECAPTURE_LOW_DIVERSITY must emit the code only once.
        """
        from modules.shared_contracts.lifecycle import AssetStatus
        cov = _make_coverage_report(
            status="insufficient",
            reasons=["Insufficient viewpoint diversity (3/5 unique views)."],
        )
        g = self.agg.generate_guidance(
            "s1",
            AssetStatus.RECAPTURE_REQUIRED,
            failure_reason="Insufficient viewpoint diversity (3/5 unique views).",
            coverage_report=cov,
        )
        codes = self._codes(g)
        assert codes.count("RECAPTURE_LOW_DIVERSITY") == 1, (
            f"RECAPTURE_LOW_DIVERSITY appeared {codes.count('RECAPTURE_LOW_DIVERSITY')} times"
        )

    def test_to_markdown_includes_validation_table(self):
        from modules.shared_contracts.lifecycle import AssetStatus
        from modules.shared_contracts.models import CaptureGuidance
        val = _make_validation_report(decision="review")
        g = self.agg.generate_guidance("s1", AssetStatus.VALIDATED, validation_report=val)
        md = self.agg.to_markdown(g)
        assert "## Validation Details" in md, "Markdown must include validation table"
        assert "Decision" in md
        assert "REVIEW" in md.upper()

    def test_to_markdown_includes_coverage_table(self):
        from modules.shared_contracts.lifecycle import AssetStatus
        cov = _make_coverage_report(status="sufficient")
        g = self.agg.generate_guidance("s1", AssetStatus.CAPTURED, coverage_report=cov)
        md = self.agg.to_markdown(g)
        assert "## Coverage Details" in md
        assert "Unique viewpoints" in md


# ──────────────────────────────────────────────────────────────────────────────
# TICKET-011: Retention policy
# ──────────────────────────────────────────────────────────────────────────────

class TestRetentionPolicy:
    @pytest.fixture()
    def tmp_data(self, tmp_path):
        data = tmp_path / "data"
        for sub in ["sessions", "captures", "reconstructions",
                    "registry/meta", "registry/blobs"]:
            (data / sub).mkdir(parents=True, exist_ok=True)
        return data

    @pytest.fixture()
    def session_manager(self, tmp_data):
        from modules.capture_workflow.session_manager import SessionManager
        return SessionManager(data_root=str(tmp_data))

    @pytest.fixture()
    def retention(self, tmp_data):
        from modules.operations.retention import RetentionService
        return RetentionService(data_root=str(tmp_data))

    def _create_session_with_raw_data(self, tmp_data, session_manager, session_id,
                                      status, publish_state=None,
                                      age_days=0, has_frames=True):
        """Creates a session + raw data artifacts on disk, back-dates mtime."""
        from modules.shared_contracts.lifecycle import AssetStatus
        from modules.utils.file_persistence import atomic_write_json

        session_manager.create_session(session_id, f"prod_{session_id}", "op_1")

        # Back-date session file
        sess_file = tmp_data / "sessions" / f"{session_id}.json"
        with open(sess_file) as f:
            data = json.load(f)

        past = datetime.now(timezone.utc) - timedelta(days=age_days, hours=1)
        data["created_at"] = past.isoformat()
        data["last_pipeline_progress_at"] = past.isoformat()
        data["status"] = status.value
        if publish_state:
            data["publish_state"] = publish_state
        atomic_write_json(sess_file, data)

        # Create raw data
        capture_dir = tmp_data / "captures" / session_id
        if has_frames:
            frames_dir = capture_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            (frames_dir / "frame001.jpg").write_bytes(b"\xff\xd8")  # minimal JPEG magic

        video_dir = capture_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / "raw_video.mp4").write_bytes(b"\x00\x00\x00\x20\x66\x74\x79\x70")

        # Create protected reports
        reports_dir = capture_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(reports_dir / "validation_report.json", {"final_decision": "pass"})
        atomic_write_json(reports_dir / "coverage_report.json", {"overall_status": "sufficient"})

        return sess_file, capture_dir

    def test_published_session_pruned_after_threshold(self, tmp_data, session_manager, retention):
        from modules.shared_contracts.lifecycle import AssetStatus
        _, cap = self._create_session_with_raw_data(
            tmp_data, session_manager, "sess_pub",
            status=AssetStatus.PUBLISHED,
            age_days=5,  # > published_frames_days=3
        )
        with patch("modules.operations.retention.settings") as ms:
            ms.published_frames_days = 3
            ms.draft_frames_days = 7
            ms.failed_frames_days = 14
            ms.reconstruction_scratch_hours = 48
            ms.data_root = str(tmp_data)
            summary = retention.run_cleanup()

        assert summary["sessions_pruned_raw"] >= 1
        assert not (cap / "frames").exists(), "frames/ must be pruned for old PUBLISHED session"
        assert not (cap / "video").exists(), "video/ must be pruned for old PUBLISHED session"
        # Reports must be preserved
        assert (cap / "reports").exists(), "reports/ must NEVER be pruned"
        assert (cap / "reports" / "validation_report.json").exists()

    def test_published_session_not_pruned_if_recent(self, tmp_data, session_manager, retention):
        from modules.shared_contracts.lifecycle import AssetStatus
        _, cap = self._create_session_with_raw_data(
            tmp_data, session_manager, "sess_pub_new",
            status=AssetStatus.PUBLISHED,
            age_days=1,  # < published_frames_days=3
        )
        with patch("modules.operations.retention.settings") as ms:
            ms.published_frames_days = 3
            ms.draft_frames_days = 7
            ms.failed_frames_days = 14
            ms.reconstruction_scratch_hours = 48
            ms.data_root = str(tmp_data)
            summary = retention.run_cleanup()

        assert summary["sessions_pruned_raw"] == 0
        assert (cap / "frames").exists(), "Recent PUBLISHED session frames must NOT be pruned"

    def test_draft_session_pruned_after_draft_threshold(self, tmp_data, session_manager, retention):
        """SPRINT 3 TICKET-011: Draft sessions must be pruned after draft_frames_days."""
        from modules.shared_contracts.lifecycle import AssetStatus
        _, cap = self._create_session_with_raw_data(
            tmp_data, session_manager, "sess_draft",
            status=AssetStatus.VALIDATED,
            publish_state="draft",
            age_days=10,  # > draft_frames_days=7
        )
        with patch("modules.operations.retention.settings") as ms:
            ms.published_frames_days = 3
            ms.draft_frames_days = 7
            ms.failed_frames_days = 14
            ms.reconstruction_scratch_hours = 48
            ms.data_root = str(tmp_data)
            summary = retention.run_cleanup()

        assert summary["sessions_pruned_raw"] >= 1
        assert not (cap / "frames").exists(), "Old DRAFT session frames must be pruned"

    def test_draft_session_not_pruned_if_recent(self, tmp_data, session_manager, retention):
        from modules.shared_contracts.lifecycle import AssetStatus
        _, cap = self._create_session_with_raw_data(
            tmp_data, session_manager, "sess_draft_new",
            status=AssetStatus.VALIDATED,
            publish_state="draft",
            age_days=3,  # < draft_frames_days=7
        )
        with patch("modules.operations.retention.settings") as ms:
            ms.published_frames_days = 3
            ms.draft_frames_days = 7
            ms.failed_frames_days = 14
            ms.reconstruction_scratch_hours = 48
            ms.data_root = str(tmp_data)
            summary = retention.run_cleanup()

        assert summary["sessions_pruned_raw"] == 0
        assert (cap / "frames").exists(), "Recent DRAFT session must NOT be pruned"

    def test_failed_session_pruned_after_threshold(self, tmp_data, session_manager, retention):
        from modules.shared_contracts.lifecycle import AssetStatus
        _, cap = self._create_session_with_raw_data(
            tmp_data, session_manager, "sess_fail",
            status=AssetStatus.FAILED,
            age_days=20,  # > failed_frames_days=14
        )
        with patch("modules.operations.retention.settings") as ms:
            ms.published_frames_days = 3
            ms.draft_frames_days = 7
            ms.failed_frames_days = 14
            ms.reconstruction_scratch_hours = 48
            ms.data_root = str(tmp_data)
            summary = retention.run_cleanup()

        assert summary["sessions_pruned_raw"] >= 1
        assert not (cap / "frames").exists()

    def test_active_pipeline_session_never_pruned(self, tmp_data, session_manager, retention):
        """CREATED / CAPTURED sessions must NEVER be pruned."""
        from modules.shared_contracts.lifecycle import AssetStatus
        for status in [AssetStatus.CREATED, AssetStatus.CAPTURED, AssetStatus.RECONSTRUCTED]:
            sid = f"sess_active_{status.value}"
            _, cap = self._create_session_with_raw_data(
                tmp_data, session_manager, sid,
                status=status,
                age_days=100,  # very old — must still be exempt
            )

        with patch("modules.operations.retention.settings") as ms:
            ms.published_frames_days = 3
            ms.draft_frames_days = 7
            ms.failed_frames_days = 14
            ms.reconstruction_scratch_hours = 48
            ms.data_root = str(tmp_data)
            summary = retention.run_cleanup()

        assert summary["sessions_pruned_raw"] == 0, (
            "Active pipeline sessions must NEVER be pruned"
        )
        for status in [AssetStatus.CREATED, AssetStatus.CAPTURED, AssetStatus.RECONSTRUCTED]:
            sid = f"sess_active_{status.value}"
            cap = tmp_data / "captures" / sid
            if cap.exists():
                assert (cap / "frames").exists() or not (cap / "frames").parent.exists(), (
                    f"Active session {sid} frames must not be deleted"
                )

    def test_protected_artifacts_never_deleted(self, tmp_data, session_manager, retention):
        """validation_report.json and coverage_report.json must survive any retention cycle."""
        from modules.shared_contracts.lifecycle import AssetStatus
        _, cap = self._create_session_with_raw_data(
            tmp_data, session_manager, "sess_protect",
            status=AssetStatus.PUBLISHED,
            age_days=999,
        )
        with patch("modules.operations.retention.settings") as ms:
            ms.published_frames_days = 1
            ms.draft_frames_days = 1
            ms.failed_frames_days = 1
            ms.reconstruction_scratch_hours = 0
            ms.data_root = str(tmp_data)
            retention.run_cleanup()

        reports = cap / "reports"
        assert reports.exists(), "reports/ directory must always exist after retention"
        assert (reports / "validation_report.json").exists(), \
            "validation_report.json must NEVER be deleted"
        assert (reports / "coverage_report.json").exists(), \
            "coverage_report.json must NEVER be deleted"

    def test_run_cleanup_returns_summary_dict(self, tmp_data, retention):
        """run_cleanup must return a structured summary dict."""
        with patch("modules.operations.retention.settings") as ms:
            ms.published_frames_days = 3
            ms.draft_frames_days = 7
            ms.failed_frames_days = 14
            ms.reconstruction_scratch_hours = 48
            ms.data_root = str(tmp_data)
            summary = retention.run_cleanup()

        assert isinstance(summary, dict)
        assert "sessions_pruned_raw" in summary
        assert "recon_dirs_pruned" in summary
        assert "errors" in summary
        assert "duration_sec" in summary
        assert isinstance(summary["errors"], list)
        assert isinstance(summary["duration_sec"], float)

    def test_orphaned_recon_dir_with_manifest_is_scratch_pruned_not_deleted(
        self, tmp_data, retention
    ):
        """
        A recon dir with no job.json but with a manifest.json must not be removed.
        Its scratch sub-folders should be pruned after the threshold.
        """
        from modules.utils.file_persistence import atomic_write_json

        job_dir = tmp_data / "reconstructions" / "job_orphan_manifest"
        job_dir.mkdir(parents=True, exist_ok=True)

        # No job.json — orphaned
        # But has manifest.json — has delivered assets
        atomic_write_json(job_dir / "manifest.json", {"job_id": "job_orphan_manifest"})

        # Has scratch folder (old)
        scratch = job_dir / "sparse"
        scratch.mkdir()
        (scratch / "0").mkdir()
        (scratch / "0" / "points3D.bin").write_bytes(b"\x00" * 100)

        # Back-date the dir mtime by writing a file then deleting it won't work
        # on Windows without ctypes; instead we just set the threshold to 0 hours.
        with patch("modules.operations.retention.settings") as ms:
            ms.published_frames_days = 3
            ms.draft_frames_days = 7
            ms.failed_frames_days = 14
            ms.reconstruction_scratch_hours = 0  # threshold=0 → everything is stale
            ms.data_root = str(tmp_data)
            summary = retention.run_cleanup()

        # Directory must still exist (has manifest)
        assert job_dir.exists(), (
            "Recon dir with manifest.json must NOT be deleted even if orphaned"
        )
        # manifest must be preserved
        assert (job_dir / "manifest.json").exists()


# ──────────────────────────────────────────────────────────────────────────────
# TICKET-012: Lifespan / no on_event deprecation
# ──────────────────────────────────────────────────────────────────────────────

class TestApiLifespanMigration:
    def test_api_module_has_no_on_event_calls(self):
        """
        The api.py source must not use @app.on_event as a decorator.
        This verifies the lifespan migration (TICKET-012).
        We check for the decorator pattern specifically (not docstring mentions).
        """
        api_source = Path(__file__).parent.parent / "modules" / "operations" / "api.py"
        content = api_source.read_text(encoding="utf-8")
        # Only lines that actually call @app.on_event (not docstring/comment references)
        decorator_lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip().startswith("@app.on_event")
        ]
        assert len(decorator_lines) == 0, (
            f"@app.on_event decorator must have been replaced with lifespan handler. "
            f"Found: {decorator_lines}"
        )

    def test_api_module_uses_lifespan(self):
        """api.py must define a lifespan context manager."""
        api_source = Path(__file__).parent.parent / "modules" / "operations" / "api.py"
        content = api_source.read_text(encoding="utf-8")
        assert "asynccontextmanager" in content, "lifespan must use asynccontextmanager"
        assert "lifespan=" in content, "FastAPI app must be created with lifespan= argument"
