import pytest
from modules.qa_validation.validator import AssetValidator
from modules.qa_validation.rules import ValidationThresholds

def test_validator_pass():
    validator = AssetValidator()
    asset_data = {
        "poly_count": 10_000,
        "texture_status": "complete",
        "bbox": {"width": 10.0, "height": 20.0, "depth": 5.0},
        "ground_offset": 0.5
    }
    
    report = validator.validate("test_id", asset_data)
    
    assert report.final_decision == "pass"
    assert report.mobile_performance_grade == "A"
    assert report.bbox_reasonable is True
    assert report.ground_aligned is True

def test_validator_fail_polycount():
    validator = AssetValidator()
    asset_data = {
        "poly_count": 150_000,
        "texture_status": "complete",
        "bbox": {"width": 10.0, "height": 20.0, "depth": 5.0},
        "ground_offset": 0.5
    }
    
    report = validator.validate("test_id", asset_data)
    
    assert report.final_decision == "fail"
    assert report.mobile_performance_grade == "D"

def test_validator_review_polycount():
    validator = AssetValidator()
    asset_data = {
        "poly_count": 75_000,
        "texture_status": "complete",
        "bbox": {"width": 10.0, "height": 20.0, "depth": 5.0},
        "ground_offset": 0.5
    }
    
    report = validator.validate("test_id", asset_data)
    
    assert report.final_decision == "review"
    assert report.mobile_performance_grade == "C"

def test_validator_custom_thresholds():
    custom_thresholds = ValidationThresholds(polycount_pass=10_000, polycount_review=20_000)
    validator = AssetValidator(thresholds=custom_thresholds)
    
    asset_data = {
        "poly_count": 15_000,
        "texture_status": "complete",
        "bbox": {"width": 10.0, "height": 20.0, "depth": 5.0},
        "ground_offset": 0.5
    }
    
    report = validator.validate("test_id", asset_data)
    
    assert report.final_decision == "review"

def test_validator_fail_texture():
    validator = AssetValidator()
    asset_data = {
        "poly_count": 10_000,
        "texture_status": "missing_critical",
        "bbox": {"width": 10.0, "height": 20.0, "depth": 5.0},
        "ground_offset": 0.5
    }
    
    report = validator.validate("test_id", asset_data)
    
    assert report.final_decision == "fail"
