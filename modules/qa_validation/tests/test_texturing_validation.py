import pytest
from modules.qa_validation.validator import AssetValidator
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
from modules.integration_flow import IntegrationFlow

def test_textured_glb_overrides_stale_cleanup_stats():
    # GIVEN: Stale cleanup_stats indicating no UVs
    cleanup_stats = {
        "has_uv": False,
        "has_material": False,
        "texture_integrity_status": "missing",
        "material_semantic_status": "geometry_only",
        "decimation": {
            "uv_preserved": False,
            "material_preserved": False
        },
        "delivery_profile": "mobile_high"
    }
    
    # GIVEN: Final export metrics confirming a textured GLB
    export_report = {
        "texture_count": 1,
        "material_count": 1,
        "all_textured_primitives_have_texcoord_0": True,
        "all_primitives_have_position": True,
        "all_primitives_have_normal": True,
        "texture_applied": True,
        "delivery_ready": True
    }
    
    metadata = NormalizedMetadata(
        bbox_min={"x": -10, "y": -10, "z": 0},
        bbox_max={"x": 10, "y": 10, "z": 10},
        pivot_offset={"x": 0, "y": 0, "z": 0},
        final_polycount=1000
    )
    
    # WHEN: Mapping to validator input
    validator_input = IntegrationFlow.map_metadata_to_validator_input(
        metadata=metadata,
        cleanup_stats=cleanup_stats,
        export_report=export_report
    )
    
    # THEN: Validation input should reflect the final truth
    assert validator_input["has_uv"] is True
    assert validator_input["has_material"] is True
    assert validator_input["texture_integrity_status"] == "complete"
    assert validator_input["material_semantic_status"] == "diffuse_textured"
    
    # WHEN: Validating
    validator = AssetValidator()
    report = validator.validate("test_asset", validator_input)
    
    # THEN: Report should pass texture checks
    assert report.texture_status == "complete"
    assert report.material_semantic_status == "diffuse_textured"
    assert "texture_uv_integrity" in report.passed_checks or any("texture" in c for c in report.passed_checks)
    # Check if decimation failure was ignored due to texturing
    assert "decimation" in report.passed_checks

def test_black_atlas_fails_qa():
    # GIVEN: A mock texture result with high black ratio
    quality_stats = {
        "texture_quality_status": "fail",
        "texture_quality_reasons": ["High black pixel ratio: 0.90 > 0.4"],
        "black_pixel_ratio": 0.9,
        "near_black_ratio": 0.95,
        "average_luminance": 0.05,
        "expected_product_color_match_score": 0.1
    }
    
    asset_data = {
        "poly_count": 1000,
        "delivery_profile": "mobile_high",
        "texture_integrity_status": "complete",
        "material_semantic_status": "diffuse_textured",
        "has_uv": True,
        "has_material": True,
        "texture_count": 1,
        "all_textured_primitives_have_texcoord_0": True,
        "texture_applied": True,
        "expected_product_color": "white_cream",
        **quality_stats
    }
    
    # WHEN: Validating
    validator = AssetValidator()
    report = validator.validate("test_black_asset", asset_data)
    
    # THEN: Final decision should be fail
    assert report.final_decision == "fail"
    assert "texture_quality" in report.blocking_checks
    assert report.texture_quality_status == "fail"

def test_neon_atlas_fails_qa():
    # GIVEN: A mock texture result with neon artifacts
    quality_stats = {
        "texture_quality_status": "fail",
        "texture_quality_reasons": ["Neon artifacts detected: 0.15"],
        "neon_artifact_ratio": 0.15
    }
    
    asset_data = {
        "poly_count": 1000,
        "delivery_profile": "mobile_high",
        "texture_integrity_status": "complete",
        "material_semantic_status": "diffuse_textured",
        "has_uv": True,
        "has_material": True,
        **quality_stats
    }
    
    validator = AssetValidator()
    report = validator.validate("test_neon_asset", asset_data)
    
    assert report.final_decision == "fail"
    assert "texture_quality" in report.blocking_checks
