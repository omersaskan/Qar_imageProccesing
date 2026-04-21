import pytest
from modules.qa_validation.validator import AssetValidator
from modules.qa_validation.rules import ValidationThresholds


def make_asset_data(**overrides):
    asset_data = {
        "poly_count": 10_000,
        "texture_status": "complete",
        "bbox": {"width": 10.0, "height": 20.0, "depth": 5.0},
        "ground_offset": 0.5,
        "cleanup_stats": {
            "isolation": {
                "component_count": 1,
                "initial_faces": 1000,
                "final_faces": 950,
                "removed_plane_face_share": 0.0,
                "removed_plane_vertex_ratio": 0.0,
                "compactness_score": 0.5,
                "selected_component_score": 0.8,
            }
        },
        "texture_path_exists": True,
        "has_uv": True,
        "has_material": True,
        "texture_applied_successfully": True,
        "texture_integrity_status": "complete",
        "material_integrity_status": "complete",
        "material_semantic_status": "diffuse_textured",
        "has_embedded_texture": True,
        "texture_count": 1,
        "material_count": 1,
        "delivery_geometry_count": 1,
        "delivery_component_count": 1,
    }
    asset_data.update(overrides)
    # Synchronize legacy texture_status override with modern texture_integrity_status
    if "texture_status" in overrides:
        asset_data["texture_integrity_status"] = overrides["texture_status"]
    return asset_data


def test_validator_pass():
    validator = AssetValidator()
    asset_data = make_asset_data()
    
    report = validator.validate("test_id", asset_data)
    
    assert report.final_decision == "pass"
    assert report.mobile_performance_grade == "A"
    assert report.bbox_reasonable is True
    assert report.ground_aligned is True

def test_validator_fail_polycount():
    validator = AssetValidator()
    asset_data = make_asset_data(poly_count=150_000)
    
    report = validator.validate("test_id", asset_data)
    
    assert report.final_decision == "fail"
    assert report.mobile_performance_grade == "D"

def test_validator_review_polycount():
    validator = AssetValidator()
    asset_data = make_asset_data(poly_count=75_000)
    
    report = validator.validate("test_id", asset_data)
    
    assert report.final_decision == "review"
    assert report.mobile_performance_grade == "C"

def test_validator_custom_thresholds():
    custom_thresholds = ValidationThresholds(polycount_pass=10_000, polycount_review=20_000)
    validator = AssetValidator(thresholds=custom_thresholds)
    
    asset_data = make_asset_data(poly_count=15_000)
    
    report = validator.validate("test_id", asset_data)
    
    assert report.final_decision == "review"

def test_validator_fail_texture():
    validator = AssetValidator()
    asset_data = make_asset_data(texture_status="missing_critical")
    
    report = validator.validate("test_id", asset_data)
    
    assert report.final_decision == "fail"
