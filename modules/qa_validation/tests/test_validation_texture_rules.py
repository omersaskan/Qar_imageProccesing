import pytest
from modules.qa_validation.rules import (
    validate_texture,
    validate_texture_integrity,
    ValidationThresholds,
)
from modules.qa_validation.validator import AssetValidator


def test_validate_texture_honesty():
    assert validate_texture("complete") == "pass"
    assert validate_texture("degraded") == "review"
    assert validate_texture("minor_missing") == "review"
    assert validate_texture("missing") == "fail"


def test_validate_texture_integrity_missing():
    thresholds = ValidationThresholds()
    data = {
        "texture_integrity_status": "missing",
        "has_uv": False,
        "has_material": False,
        "texture_count": 0
    }
    res = validate_texture_integrity(data, thresholds)
    assert res["texture_uv_integrity"] == "fail"
    assert res["texture_application"] == "fail"
    assert res["material_integrity"] == "fail"


def test_validate_texture_integrity_degraded_uv_only():
    thresholds = ValidationThresholds()
    data = {
        "texture_integrity_status": "degraded",
        "has_uv": True,
        "has_material": False,
        "texture_count": 0
    }
    res = validate_texture_integrity(data, thresholds)
    assert res["texture_uv_integrity"] == "pass"
    assert res["texture_application"] == "review"
    assert res["material_integrity"] == "review"


def test_validate_texture_integrity_complete():
    thresholds = ValidationThresholds()
    data = {
        "texture_integrity_status": "complete",
        "has_uv": True,
        "has_material": True,
        "texture_count": 1
    }
    res = validate_texture_integrity(data, thresholds)
    assert res["texture_uv_integrity"] == "pass"
    assert res["texture_application"] == "pass"
    assert res["material_integrity"] == "pass"


def test_validator_overall_decision_for_texture_states():
    v = AssetValidator()
    
    # Degraded status should yield "review"
    data_degraded = {
        "poly_count": 10000,
        "bbox": {"x": 10, "y": 10, "z": 10},
        "ground_offset": 0.0,
        "cleanup_stats": {
            "isolation": {
                "final_faces": 100, 
                "initial_faces": 100, 
                "component_count": 1,
                "compactness_score": 0.5,
                "selected_component_score": 0.8
            }
        },
        "texture_integrity_status": "degraded",
        "material_semantic_status": "uv_only",
        "has_uv": True,
        "has_material": False,
        "texture_count": 0,
        "delivery_geometry_count": 1,
        "delivery_component_count": 1,
    }
    rep_degraded = v.validate("asset_1", data_degraded)
    assert rep_degraded.final_decision == "review"
    
    # Missing status should yield "fail"
    data_missing = data_degraded.copy()
    data_missing["texture_integrity_status"] = "missing"
    data_missing["has_uv"] = False
    
    rep_missing = v.validate("asset_2", data_missing)
    assert rep_missing.final_decision == "fail"
    
    # Complete status should yield "pass"
    data_complete = data_degraded.copy()
    data_complete["texture_integrity_status"] = "complete"
    data_complete["material_semantic_status"] = "pbr_complete"
    data_complete["has_uv"] = True
    data_complete["has_material"] = True
    data_complete["texture_count"] = 3 # baseColor, normal, MR
    
    rep_complete = v.validate("asset_3", data_complete)
    assert rep_complete.final_decision == "pass"
    assert rep_complete.material_quality_grade == "S"
    assert rep_complete.material_semantic_status == "pbr_complete"

def test_validator_diffuse_grading():
    v = AssetValidator()
    data = {
        "poly_count": 10000,
        "bbox": {"x": 10, "y": 10, "z": 10},
        "ground_offset": 0.0,
        "cleanup_stats": {
            "isolation": {
                "final_faces": 100, 
                "initial_faces": 100, 
                "component_count": 1,
                "compactness_score": 0.5,
                "selected_component_score": 0.8
            }
        },
        "texture_integrity_status": "complete",
        "material_semantic_status": "diffuse_textured",
        "has_uv": True,
        "has_material": True,
        "texture_count": 1,
        "delivery_geometry_count": 1,
        "delivery_component_count": 1,
    }
    rep = v.validate("asset_diffuse", data)
    assert rep.final_decision == "pass"
    assert rep.material_quality_grade == "B"
    assert rep.material_semantic_status == "diffuse_textured"
