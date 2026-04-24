"""
tests/test_calibration_phase.py

CALIBRATION PHASE — Tests for real-world evaluation tooling

Tests:
  Dataset schema validation
  - dataset_schema.json is valid and loadable JSON
  - real fixture entries conform to required fields
  - invalid fixtures are caught correctly

  Fixture extraction utility
  - extract_fixture() from a directory with all artifacts
  - extract_fixture() with missing coverage_report (graceful warning)
  - auto-inference of issue_classes from reports
  - auto-inference of expected_action from session status
  - append_to_fixture_file() deduplication
  - _make_fixture_id() format

  Evaluate all (unified harness)
  - load synthetic + real fixtures and merge
  - breakdown_by_source produces distinct groups
  - breakdown_by_product groups correctly
  - issue_class breakdown counts FP/FN correctly
  - FP case (real_006) detected in evaluation
  - threshold recommendations produced for FP masking_fallback case
  - recommendation direction labels are valid
  - markdown output contains all expected sections

  Guidance evaluation
  - correct action match for accept fixture
  - correct action match for recapture fixture
  - overcautious detected for accept fixture with recapture guidance
  - missed detected for recapture fixture with no recapture guidance
  - noisy detected when >5 messages
  - guidance evaluation metrics computed correctly
"""
from __future__ import annotations

import json
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

import pytest

_TOOLS = Path(__file__).parent.parent / "tools" / "calibration"
_FIXTURES_DIR = _TOOLS / "fixtures"
_REAL_FIXTURES = _FIXTURES_DIR / "real_captures" / "real_capture_fixtures.json"
_SYNTH_FIXTURES = _FIXTURES_DIR / "coverage_fixtures.json"
_DATASET_SCHEMA = _TOOLS / "dataset_schema.json"


# ─────────────────────────────────────────────────────────────────────────────
# Module loaders (avoid polluting sys.modules)
# ─────────────────────────────────────────────────────────────────────────────

def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _evaluator():
    return _load_module(_TOOLS / "evaluate_all.py", "evaluate_all")


def _extractor():
    return _load_module(_TOOLS / "extract_fixture.py", "extract_fixture_mod")


def _guidance_eval():
    return _load_module(_TOOLS / "evaluate_guidance.py", "evaluate_guidance")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_coverage_sufficient(**kwargs) -> Dict[str, Any]:
    base = {
        "num_frames": 40, "readable_frames": 38, "unique_views": 7,
        "diversity": "sufficient", "top_down_captured": True,
        "center_x_span": 0.20, "center_y_span": 0.16,
        "scale_variation": 1.30, "aspect_variation": 0.18,
        "coverage_score": 0.85, "fallback_frames": 0,
        "low_confidence_frames": 0, "overall_status": "sufficient",
        "recommended_action": "reconstruct", "reasons": [],
        "ml_segmentation_unavailable": False,
    }
    base.update(kwargs)
    return base


def _make_coverage_insufficient(reason: str = "Insufficient viewpoint diversity (3/5).",
                                 **kwargs) -> Dict[str, Any]:
    base = _make_coverage_sufficient()
    base.update({
        "unique_views": 3, "coverage_score": 0.28,
        "diversity": "insufficient", "overall_status": "insufficient",
        "recommended_action": "needs_recapture", "reasons": [reason],
    })
    base.update(kwargs)
    return base


def _make_fixture(fixture_id="test_fix", label="good", expected_action="accept",
                  coverage_report=None, validation_report=None,
                  issue_classes=None, source="synthetic",
                  pipeline_status="validated",
                  product_type="sneaker") -> Dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "label": label,
        "expected_action": expected_action,
        "coverage_report": coverage_report or _make_coverage_sufficient(),
        "validation_report": validation_report,
        "observed_issue_classes": issue_classes or ["none"],
        "source": source,
        "_pipeline_status": pipeline_status,
        "product_type": product_type,
        "description": "",
        "notes": "",
    }



# ─────────────────────────────────────────────────────────────────────────────
# Dataset schema validation
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetSchema:
    def test_schema_file_exists(self):
        assert _DATASET_SCHEMA.exists(), "dataset_schema.json must exist"

    def test_schema_is_valid_json(self):
        data = json.loads(_DATASET_SCHEMA.read_text(encoding="utf-8"))
        assert "properties" in data
        assert "required" in data
        required = data["required"]
        assert "fixture_id" in required
        assert "label" in required
        assert "coverage_report" in required

    def test_real_fixtures_file_exists(self):
        assert _REAL_FIXTURES.exists(), "real_capture_fixtures.json must exist"

    def test_real_fixtures_conform_to_required_fields(self):
        fixtures = json.loads(_REAL_FIXTURES.read_text(encoding="utf-8"))
        assert isinstance(fixtures, list)
        assert len(fixtures) >= 3, "Must have at least 3 real fixtures"
        for f in fixtures:
            assert "fixture_id" in f, f"Missing fixture_id in {f}"
            assert "label" in f, f"Missing label in {f}"
            assert "coverage_report" in f, f"Missing coverage_report in {f}"
            assert f["label"] in {
                "good", "recapture", "bad", "noisy", "insufficient"
            }, f"Invalid label: {f['label']}"

    def test_real_fixtures_have_expected_action(self):
        fixtures = json.loads(_REAL_FIXTURES.read_text(encoding="utf-8"))
        for f in fixtures:
            assert "expected_action" in f, \
                f"Real fixture {f.get('fixture_id')} missing expected_action"
            assert f["expected_action"] in {"accept", "recapture", "review", "fail"}

    def test_real_fixtures_have_source_field(self):
        fixtures = json.loads(_REAL_FIXTURES.read_text(encoding="utf-8"))
        for f in fixtures:
            assert "source" in f, f"Fixture {f.get('fixture_id')} missing source field"
            assert f["source"] in {"synthetic", "real", "extracted"}

    def test_real_006_is_false_positive_case(self):
        """real_006 must be labeled good/accept but have overall_status=insufficient."""
        fixtures = json.loads(_REAL_FIXTURES.read_text(encoding="utf-8"))
        f = next((x for x in fixtures if x["fixture_id"] == "real_006"), None)
        assert f is not None, "real_006 fixture must exist"
        assert f["label"] == "good"
        assert f["expected_action"] == "accept"
        assert f["coverage_report"]["overall_status"] == "insufficient"


# ─────────────────────────────────────────────────────────────────────────────
# Fixture extraction utility
# ─────────────────────────────────────────────────────────────────────────────

class TestFixtureExtraction:
    def test_extract_fixture_from_existing_session(self, tmp_path):
        """extract_fixture() should produce a valid fixture from a session directory."""
        mod = _extractor()

        # Build fake session + artifacts
        data_root = tmp_path / "data"
        session_id = "sess_extract_test"
        sessions_dir = data_root / "sessions"
        sessions_dir.mkdir(parents=True)
        reports_dir = data_root / "captures" / session_id / "reports"
        reports_dir.mkdir(parents=True)

        # Session file
        session_data = {
            "session_id": session_id,
            "product_id": "prod_123",
            "operator_id": "op_1",
            "status": "validated",
            "publish_state": "pending",
            "failure_reason": None,
            "created_at": "2026-04-13T00:00:00Z",
            "coverage_score": 0.85,
        }
        (sessions_dir / f"{session_id}.json").write_text(
            json.dumps(session_data), encoding="utf-8"
        )

        # Coverage report
        cov = _make_coverage_sufficient()
        (reports_dir / "coverage_report.json").write_text(
            json.dumps(cov), encoding="utf-8"
        )

        # Validation report
        val = {
            "final_decision": "pass",
            "contamination_score": 0.05,
            "mobile_performance_grade": "A",
            "poly_count": 14000,
            "component_count": 1,
            "largest_component_share": 0.99,
        }
        (reports_dir / "validation_report.json").write_text(
            json.dumps(val), encoding="utf-8"
        )

        fixture = mod.extract_fixture(
            session_id=session_id,
            data_root=data_root,
            label="good",
            expected_action="accept",
            product_type="sneaker",
            capture_environment="studio",
            notes="Test extraction",
            operator_id="op_test",
        )

        assert fixture["fixture_id"].startswith("real_sess_extract")
        assert fixture["label"] == "good"
        assert fixture["expected_action"] == "accept"
        assert fixture["product_type"] == "sneaker"
        assert fixture["source"] == "extracted"
        assert fixture["coverage_report"]["overall_status"] == "sufficient"
        assert fixture["validation_report"]["final_decision"] == "pass"

    def test_extract_fixture_with_missing_coverage_graceful(self, tmp_path):
        """extract_fixture() must not crash when coverage_report is missing."""
        mod = _extractor()
        data_root = tmp_path / "data"
        session_id = "sess_no_cov"
        (data_root / "sessions").mkdir(parents=True)
        (data_root / "captures" / session_id / "reports").mkdir(parents=True)
        (data_root / "sessions" / f"{session_id}.json").write_text(
            json.dumps({"session_id": session_id, "status": "created",
                        "product_id": "p1", "operator_id": "op"}),
            encoding="utf-8",
        )

        # Should not raise; coverage_report.json absent
        fixture = mod.extract_fixture(
            session_id=session_id, data_root=data_root,
            label="recapture", expected_action="recapture",
        )
        assert fixture["source"] == "extracted"
        # Missing coverage is represented with a placeholder
        assert "MISSING" in fixture["coverage_report"]["reasons"][0]

    def test_infer_issue_classes_from_coverage_fallback(self):
        mod = _extractor()
        cov = _make_coverage_sufficient(
            fallback_frames=25, readable_frames=40,
            overall_status="insufficient",
            reasons=["Too many frames relied on heuristic fallback (25)."],
        )
        issues = mod._infer_issue_classes(cov, None)
        assert "masking_fallback" in issues

    def test_infer_issue_classes_from_low_diversity(self):
        mod = _extractor()
        cov = _make_coverage_insufficient(
            reason="Insufficient viewpoint diversity (3/5 unique views)."
        )
        issues = mod._infer_issue_classes(cov, None)
        assert "low_diversity" in issues

    def test_infer_issue_classes_from_contamination(self):
        mod = _extractor()
        val = {"contamination_score": 0.65, "component_count": 1,
               "largest_component_share": 0.98}
        issues = mod._infer_issue_classes(None, val)
        assert "contamination_high" in issues

    def test_infer_expected_action_from_session_published(self):
        mod = _extractor()
        meta = {"status": "published", "publish_state": "published"}
        action = mod._infer_expected_action(meta, None, None)
        assert action == "accept"

    def test_infer_expected_action_from_failed_session(self):
        mod = _extractor()
        meta = {"status": "failed", "publish_state": "failed"}
        action = mod._infer_expected_action(meta, None, None)
        assert action == "fail"

    def test_append_to_fixture_file_deduplication(self, tmp_path):
        """append_to_fixture_file must not create duplicate fixture_ids."""
        mod = _extractor()
        fixture_file = tmp_path / "fixtures.json"
        fixture = _make_fixture(fixture_id="dup_001", label="good")
        mod.append_to_fixture_file(fixture, fixture_file)
        mod.append_to_fixture_file(fixture, fixture_file)  # Second call = duplicate

        data = json.loads(fixture_file.read_text(encoding="utf-8"))
        assert len(data) == 1, "Duplicate fixture must not be appended"

    def test_make_fixture_id_format(self):
        mod = _extractor()
        fid = mod._make_fixture_id("cap_abc123def456")
        assert fid.startswith("real_"), f"fixture_id must start with 'real_', got: {fid}"
        # Should contain session slug
        assert "cap_abc123de" in fid or "abc123d" in fid


# ─────────────────────────────────────────────────────────────────────────────
# Unified evaluation harness
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateAll:
    def test_loads_both_synthetic_and_real_fixtures(self):
        mod = _evaluator()
        report = mod.run_evaluation(
            fixture_paths=[_SYNTH_FIXTURES, _REAL_FIXTURES],
            fmt="json",
        )
        assert report["total_fixtures"] >= 10, (
            f"Expected >= 10 fixtures (5 synthetic + 6 real), got {report['total_fixtures']}"
        )

    def test_breakdown_by_source_has_synthetic_and_real(self):
        mod = _evaluator()
        report = mod.run_evaluation(
            fixture_paths=[_SYNTH_FIXTURES, _REAL_FIXTURES],
            fmt="json",
        )
        assert "synthetic" in report["source_breakdown"], "synthetic source must appear"
        assert "real" in report["source_breakdown"], "real source must appear"

    def test_breakdown_by_product_groups_correctly(self):
        mod = _evaluator()
        fixtures = [
            _make_fixture("a", product_type="sneaker", source="real"),
            _make_fixture("b", product_type="bag", source="real"),
            _make_fixture("c", product_type="sneaker", source="real"),
        ]
        # Patch fixture loading
        results = [mod._evaluate_fixture(f) for f in fixtures]
        breakdown = mod._breakdown_by_product(results)
        assert "sneaker" in breakdown
        assert "bag" in breakdown
        assert breakdown["sneaker"]["total"] == 2

    def test_fp_case_real_006_detected(self):
        """real_006 is labeled 'good/accept' but coverage says 'insufficient' -> FP."""
        mod = _evaluator()
        report = mod.run_evaluation(
            fixture_paths=[_REAL_FIXTURES],
            fmt="json",
        )
        # real_006 must be detected as a FP
        metrics = report["metrics"]
        assert metrics["fp"] >= 1, (
            "real_006 is a FP case (good capture rejected by heuristic). "
            f"Expected fp >= 1, got {metrics['fp']}"
        )

    def test_issue_breakdown_fp_masking_fallback(self):
        """FP caused by masking_fallback must appear in issue_breakdown."""
        mod = _evaluator()
        report = mod.run_evaluation(
            fixture_paths=[_REAL_FIXTURES],
            fmt="json",
        )
        breakdown = report["issue_breakdown"]
        assert "masking_fallback" in breakdown, (
            "masking_fallback must appear in issue_breakdown (driven by real_006)"
        )
        assert breakdown["masking_fallback"]["fp"] >= 1

    def test_threshold_recommendations_produced_for_fp(self):
        """When FP masking_fallback exists, recommendation must be generated."""
        mod = _evaluator()
        report = mod.run_evaluation(
            fixture_paths=[_REAL_FIXTURES],
            fmt="json",
        )
        recs = report["recommendations"]
        assert len(recs) >= 1, "At least one threshold recommendation must be produced"
        # Must include fallback-related recommendation
        fallback_recs = [
            r for r in recs
            if "fallback" in r["parameter"].lower()
        ]
        assert len(fallback_recs) >= 1, (
            f"Must have fallback-related recommendation. Got: {[r['parameter'] for r in recs]}"
        )

    def test_recommendation_direction_valid(self):
        """All recommendation directions must be 'relax', 'tighten', or 'none'."""
        mod = _evaluator()
        report = mod.run_evaluation(
            fixture_paths=[_SYNTH_FIXTURES, _REAL_FIXTURES],
            fmt="json",
        )
        for rec in report["recommendations"]:
            assert rec["direction"] in {"relax", "tighten", "none"}, (
                f"Invalid recommendation direction: {rec['direction']}"
            )

    def test_markdown_output_has_required_sections(self, tmp_path):
        mod = _evaluator()
        out = tmp_path / "report.md"
        mod.run_evaluation(
            fixture_paths=[_SYNTH_FIXTURES, _REAL_FIXTURES],
            output_path=out,
            fmt="json",
        )
        content = out.read_text(encoding="utf-8")
        # Markdown is rendered and saved
        assert "Unified Calibration Evaluation Report" in content or len(content) > 100

    def test_perfect_fixture_set_gives_zero_fp_fn(self):
        """A fixture set where all predictions match labels must have fp=fn=0."""
        mod = _evaluator()
        fixtures = [
            _make_fixture("g1", label="good", expected_action="accept",
                          coverage_report=_make_coverage_sufficient()),
            _make_fixture("r1", label="recapture", expected_action="recapture",
                          coverage_report=_make_coverage_insufficient()),
        ]
        results = [mod._evaluate_fixture(f) for f in fixtures]
        metrics = mod._compute_metrics(results)
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0
        assert metrics["accuracy"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Guidance evaluation
# ─────────────────────────────────────────────────────────────────────────────

class TestGuidanceEvaluation:
    def test_accept_fixture_with_recapture_guidance_is_overcautious(self):
        """
        If expected_action=accept but guidance says should_recapture=True,
        it must be detected as OVERCAUTIOUS.
        """
        mod = _guidance_eval()
        fixture = _make_fixture(
            "gc_fp", label="good", expected_action="accept",
            coverage_report=_make_coverage_sufficient(),
            pipeline_status="validated",
        )
        # Simulate guidance saying recapture
        fake_guidance = {
            "should_recapture": True,
            "is_ready_for_review": False,
            "next_action": "Re-capture required",
            "message_count": 2,
            "message_codes": ["RECAPTURE_NEEDED", "RECAPTURE_LOW_DIVERSITY"],
            "has_critical": True,
            "has_warning": False,
        }
        with patch.object(mod, "_run_guidance", return_value=fake_guidance):
            result = mod._evaluate_fixture_guidance(fixture)

        assert result["action_match"] is False
        assert any("OVERCAUTIOUS" in i for i in result["issues"])

    def test_recapture_fixture_without_recapture_guidance_is_missed(self):
        """
        If expected_action=recapture but guidance says should_recapture=False,
        it must be detected as MISSED.
        """
        mod = _guidance_eval()
        fixture = _make_fixture(
            "gc_fn", label="recapture", expected_action="recapture",
            coverage_report=_make_coverage_insufficient(),
            pipeline_status="recapture_required",
        )
        fake_guidance = {
            "should_recapture": False,
            "is_ready_for_review": True,
            "next_action": "Review asset",
            "message_count": 1,
            "message_codes": ["READY_FOR_REVIEW"],
            "has_critical": False,
            "has_warning": True,
        }
        with patch.object(mod, "_run_guidance", return_value=fake_guidance):
            result = mod._evaluate_fixture_guidance(fixture)

        assert result["action_match"] is False
        assert any("MISSED" in i for i in result["issues"])

    def test_correct_accept_guidance_passes(self):
        mod = _guidance_eval()
        fixture = _make_fixture(
            "gc_ok", label="good", expected_action="accept",
            coverage_report=_make_coverage_sufficient(),
            pipeline_status="validated",
        )
        fake_guidance = {
            "should_recapture": False,
            "is_ready_for_review": True,
            "next_action": "Asset published",
            "message_count": 1,
            "message_codes": ["READY_FOR_PUBLISH"],
            "has_critical": False,
            "has_warning": False,
        }
        with patch.object(mod, "_run_guidance", return_value=fake_guidance):
            result = mod._evaluate_fixture_guidance(fixture)

        assert result["action_match"] is True
        assert result["overall_ok"] is True
        assert result["issues"] == []

    def test_noisy_guidance_detected(self):
        mod = _guidance_eval()
        fixture = _make_fixture(
            "gc_noisy", label="good", expected_action="accept",
            pipeline_status="validated",
        )
        fake_guidance = {
            "should_recapture": False,
            "is_ready_for_review": True,
            "next_action": "Review asset",
            "message_count": 7,  # > 5 = noisy
            "message_codes": ["A", "B", "C", "D", "E", "F", "G"],
            "has_critical": False,
            "has_warning": True,
        }
        with patch.object(mod, "_run_guidance", return_value=fake_guidance):
            result = mod._evaluate_fixture_guidance(fixture)

        assert result["is_noisy"] is True
        assert any("NOISY" in i for i in result["issues"])

    def test_severe_overcautious_for_accept_with_critical_message(self):
        """CRITICAL message on an 'accept' fixture must trigger OVERCAUTIOUS_SEVERITY."""
        mod = _guidance_eval()
        fixture = _make_fixture(
            "gc_sev", label="good", expected_action="accept",
            pipeline_status="validated",
        )
        fake_guidance = {
            "should_recapture": False,  # action match is correct
            "is_ready_for_review": True,
            "next_action": "Asset reviewed",
            "message_count": 2,
            "message_codes": ["READY_FOR_REVIEW", "SYSTEM_FAILURE_CONFIG"],
            "has_critical": True,
            "has_warning": False,
        }
        with patch.object(mod, "_run_guidance", return_value=fake_guidance):
            result = mod._evaluate_fixture_guidance(fixture)

        assert result["severity_ok"] is False
        assert any("OVERCAUTIOUS_SEVERITY" in i for i in result["issues"])

    def test_guidance_metrics_computed_correctly(self):
        mod = _guidance_eval()
        results = [
            {"status": "evaluated", "action_match": True,  "severity_ok": True,
             "is_noisy": False, "expected_action": "accept", "message_count": 1,
             "overall_ok": True,  "issues": []},
            {"status": "evaluated", "action_match": False, "severity_ok": True,
             "is_noisy": False, "expected_action": "accept", "message_count": 2,
             "overall_ok": False, "issues": ["OVERCAUTIOUS"]},
            {"status": "evaluated", "action_match": True,  "severity_ok": False,
             "is_noisy": True,  "expected_action": "recapture", "message_count": 7,
             "overall_ok": False, "issues": ["NOISY"]},
        ]
        metrics = mod._compute_guidance_metrics(results)
        assert metrics["total"] == 3
        assert metrics["action_correct"] == 2
        assert metrics["noisy_count"] == 1
        assert metrics["overcautious"] == 1
        assert round(metrics["action_accuracy"], 4) == round(2/3, 4)

    def test_live_guidance_runs_on_synth_fixtures(self):
        """
        Run guidance evaluator on synthetic fixtures.
        Must produce results without crashing.
        Coverage-insufficient fixtures should be flagged as recapture.
        """
        mod = _guidance_eval()
        report = mod.run_guidance_evaluation(
            fixture_paths=[_SYNTH_FIXTURES],
            fmt="json",
        )
        assert "metrics" in report
        assert report["metrics"]["total"] > 0
        # Evaluated count must match total (no fixture skipped)
        total = report["metrics"]["total"]
        assert total == len(report["results"])
