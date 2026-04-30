import pytest
import json
from modules.shared_contracts.models import ValidationReport
from datetime import datetime, timezone

def test_validation_report_extra_forbid():
    data = {
        "asset_id": "test_asset",
        "poly_count": 1000,
        "texture_status": "complete",
        "bbox_reasonable": True,
        "ground_aligned": True,
        "mobile_performance_grade": "A",
        "final_decision": "pass",
        "extra_field": "should_be_stripped"
    }
    
    # Should not raise error because of our strip_extra_fields validator
    report = ValidationReport.model_validate(data)
    assert report.asset_id == "test_asset"
    assert not hasattr(report, "extra_field")
    
    # But direct dict initialization with extra fields (not via model_validate) 
    # might still be blocked if pydantic doesn't use the validator there.
    # Actually model_validate uses the before validator.
    
def test_validation_report_serialization():
    report = ValidationReport(
        asset_id="test_asset",
        poly_count=1000,
        texture_status="complete",
        bbox_reasonable=True,
        ground_aligned=True,
        mobile_performance_grade="A",
        final_decision="pass"
    )
    
    dump = report.model_dump()
    assert dump["asset_id"] == "test_asset"
    assert "is_mobile_ready" in dump
    assert dump["delivery_status"] == "pending"

def test_backward_compatibility_stripping():
    # Simulate an old report with legacy fields
    legacy_data = {
        "asset_id": "old_asset",
        "poly_count": 5000,
        "texture_status": "complete",
        "bbox_reasonable": True,
        "ground_aligned": True,
        "mobile_performance_grade": "B",
        "final_decision": "fail",
        "legacy_score_v1": 0.85,
        "experimental_metrics": {"noise": 0.1}
    }
    
    report = ValidationReport.model_validate(legacy_data)
    assert report.asset_id == "old_asset"
    # Verify no validation error was raised despite extra fields
    
    dump = report.model_dump()
    assert "legacy_score_v1" not in dump
    assert "experimental_metrics" not in dump
