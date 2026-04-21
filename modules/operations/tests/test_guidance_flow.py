import pytest
from unittest.mock import MagicMock
from pathlib import Path
import json

from modules.shared_contracts.models import AssetStatus, GuidanceSeverity
from modules.operations.guidance import GuidanceAggregator


def test_guidance_aggregator_no_reports():
    aggregator = GuidanceAggregator()
    guidance = aggregator.generate_guidance(
        session_id="test_session",
        status=AssetStatus.CREATED
    )

    assert guidance.session_id == "test_session"
    assert guidance.status == AssetStatus.CREATED
    # next_action should mention upload/yükle (Turkish or English)
    assert any(kw in guidance.next_action.lower() for kw in ("upload", "yükle", "video"))
    assert any(msg["code"] == "AWAITING_UPLOAD" for msg in guidance.messages)


def test_guidance_aggregator_insufficient_coverage():
    aggregator = GuidanceAggregator()
    coverage_report = {
        "overall_status": "insufficient",
        "diversity": "insufficient",
        "top_down_captured": False,
        "reasons": ["Insufficient viewpoint diversity", "Low top-down proxy"]
    }

    guidance = aggregator.generate_guidance(
        session_id="test_session",
        status=AssetStatus.CAPTURED,
        coverage_report=coverage_report
    )

    assert guidance.should_recapture is True
    # next_action should communicate that recapture is needed
    assert any(kw in guidance.next_action.lower() for kw in ("recapture", "yeniden", "recut", "retake", "video"))

    codes = [msg["code"] for msg in guidance.messages]
    # At least one viewpoint-diversity or top-down code should fire
    assert any(c in codes for c in (
        "LOW_DIVERSITY", "RECAPTURE_LOW_DIVERSITY", "INSUFFICIENT_VIEWPOINT_SPREAD",
        "MISSING_TOP_VIEWS",
    ))


def test_guidance_aggregator_validation_failure():
    aggregator = GuidanceAggregator()
    validation_report = {
        "final_decision": "fail",
        "contamination_score": 0.8,
        "material_semantic_status": "geometry_only",
        "contamination_report": {
            "texture_uv_integrity": "fail"
        }
    }

    guidance = aggregator.generate_guidance(
        session_id="test_session",
        status=AssetStatus.RECAPTURE_REQUIRED,
        validation_report=validation_report
    )

    assert guidance.should_recapture is True
    codes = [msg["code"] for msg in guidance.messages]
    # Check for contamination and UV failure codes (may be CONTAMINATION or CONTAMINATION_HIGH)
    assert any("CONTAMINATION" in c for c in codes)
    assert any("TEXTURE" in c for c in codes)


def test_guidance_markdown_generation():
    aggregator = GuidanceAggregator()
    guidance = aggregator.generate_guidance(
        session_id="test_session",
        status=AssetStatus.VALIDATED,
        validation_report={"final_decision": "pass", "material_semantic_status": "diffuse_textured"}
    )

    md = aggregator.to_markdown(guidance)
    # Header must mention session_id
    assert "test_session" in md
    # Status must appear
    assert "VALIDATED" in md
    # Should have some info message about the validated state
    assert any(kw in md for kw in ("READY_FOR_REVIEW", "ℹ️", "INFO", "Model", "Validation Details"))
