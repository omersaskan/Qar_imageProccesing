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
    assert "upload" in guidance.next_action.lower()
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
    assert "RECUT/RETAKE" in guidance.next_action
    
    codes = [msg["code"] for msg in guidance.messages]
    assert "LOW_DIVERSITY" in codes
    assert "MISSING_TOP_VIEWS" in codes

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
    assert "CONTAMINATION" in codes
    assert "TEXTURE_FAILURE" in codes
    assert "QUALITY_BAR_NOT_MET" in codes

def test_guidance_markdown_generation():
    aggregator = GuidanceAggregator()
    guidance = aggregator.generate_guidance(
        session_id="test_session",
        status=AssetStatus.VALIDATED,
        validation_report={"final_decision": "pass", "material_semantic_status": "diffuse_textured"}
    )
    
    md = aggregator.to_markdown(guidance)
    assert "# Capture Guidance: test_session" in md
    assert "VALIDATED" in md
    assert "ℹ️ INFO: Asset is validated" in md
