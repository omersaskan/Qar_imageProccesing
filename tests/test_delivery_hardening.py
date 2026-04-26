import pytest
import trimesh
import numpy as np
import os
from pathlib import Path
from PIL import Image
from modules.export_pipeline.glb_exporter import GLBExporter
from modules.qa_validation.validator import AssetValidator
from modules.qa_validation.rules import ValidationThresholds
from modules.qa_validation.texture_quality import TextureQualityAnalyzer

@pytest.fixture
def test_dir(tmp_path):
    out = tmp_path / "delivery_tests"
    out.mkdir()
    return out

@pytest.fixture
def dummy_texture(test_dir):
    tex_path = test_dir / "dummy_tex.png"
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    img.save(tex_path)
    return str(tex_path)

def create_uv_only_obj(path):
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]]
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=[[0,0], [1,0], [0,1]])
    mesh.export(str(path))

def test_glb_normals_and_accessors_fix(test_dir, dummy_texture):
    obj_path = test_dir / "test.obj"
    glb_path = test_dir / "test.glb"
    create_uv_only_obj(obj_path)
    
    exporter = GLBExporter()
    res = exporter.export_to_glb(str(obj_path), dummy_texture, str(glb_path))
    
    assert res["has_position_accessor"] is True
    assert res["has_normal_accessor"] is True
    assert res["has_texcoord_0_accessor"] is True
    
    # Reload and check strictly
    loaded = trimesh.load(str(glb_path), force="scene")
    mesh = list(loaded.geometry.values())[0]
    # In trimesh, if they were in the file, they are available.
    assert hasattr(mesh, "vertex_normals")
    assert len(mesh.vertex_normals) > 0
    assert hasattr(mesh.visual, "uv")
    assert len(mesh.visual.uv) > 0

def test_missing_normal_fails_textured_export(test_dir, dummy_texture, monkeypatch):
    obj_path = test_dir / "test_fail.obj"
    glb_path = test_dir / "test_fail.glb"
    create_uv_only_obj(obj_path)
    
    exporter = GLBExporter()
    
    # Monkeypatch fix_normals to do nothing to simulate failure to generate normals
    monkeypatch.setattr(trimesh.Trimesh, "fix_normals", lambda x, **kwargs: None)
    # Also need to prevent trimesh from automatically calculating them on export if possible, 
    # but our explicit check is after export logic or during inspection.
    
    # Actually, our exporter explicitly materializes them. 
    # To test the failure, we can mock inspection_result["has_normal_accessor"] = False
    
    orig_inspect = exporter.inspect_exported_asset
    def mock_inspect(path):
        res = orig_inspect(path)
        res["has_normal_accessor"] = False
        return res
    
    monkeypatch.setattr(exporter, "inspect_exported_asset", mock_inspect)
    
    with pytest.raises(ValueError, match="Textured GLB must have NORMAL accessor"):
        exporter.export_to_glb(str(obj_path), dummy_texture, str(glb_path))

def test_high_polycount_delivery_gate():
    validator = AssetValidator()
    
    # 50k: pass
    data_pass = {
        "poly_count": 40_000, 
        "texture_integrity_status": "complete", 
        "material_semantic_status": "pbr_complete",
        "texture_quality_status": "clean",
        "bbox": {"x": 10, "y": 10, "z": 10},
        "ground_offset": 0.0
    }
    report_pass = validator.validate("id1", data_pass)
    assert report_pass.final_decision == "pass"
    
    # 100k: review
    data_review = {
        "poly_count": 80_000, 
        "texture_integrity_status": "complete", 
        "material_semantic_status": "pbr_complete",
        "texture_quality_status": "clean",
        "bbox": {"x": 10, "y": 10, "z": 10},
        "ground_offset": 0.0
    }
    report_review = validator.validate("id2", data_review)
    assert report_review.final_decision == "review"
    assert "polycount" in report_review.warning_checks
    
    # 300k: fail
    data_fail = {
        "poly_count": 300_000, 
        "texture_integrity_status": "complete", 
        "material_semantic_status": "pbr_complete",
        "texture_quality_status": "clean",
        "bbox": {"x": 10, "y": 10, "z": 10},
        "ground_offset": 0.0
    }
    report_fail = validator.validate("id3", data_fail)
    assert report_fail.final_decision == "fail"
    assert "polycount" in report_fail.blocking_checks

def test_validation_explainability_consistency():
    validator = AssetValidator()
    
    # Fail case
    data_fail = {
        "poly_count": 10_000,
        "texture_integrity_status": "missing",
        "material_semantic_status": "geometry_only",
        "texture_quality_status": "clean",
        "bbox": {"x": 10, "y": 10, "z": 10},
        "ground_offset": 0.0
    }
    report = validator.validate("id_fail", data_fail)
    assert report.final_decision == "fail"
    assert len(report.blocking_checks) > 0
    assert "texture_integrity" in report.blocking_checks or "material_semantics" in report.blocking_checks
    
    # Pass case
    data_pass = {
        "poly_count": 10_000,
        "texture_integrity_status": "complete",
        "material_semantic_status": "pbr_complete",
        "bbox": {"x": 10, "y": 10, "z": 10},
        "ground_offset": 0.0,
        "texture_quality_status": "clean"
    }
    report_p = validator.validate("id_pass", data_pass)
    assert report_p.final_decision == "pass"
    assert len(report_p.blocking_checks) == 0
    assert len(report_p.passed_checks) > 0

def test_texture_atlas_qa_metrics(test_dir):
    analyzer = TextureQualityAnalyzer()
    
    # Create image with specific patterns
    # 100x100 image
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[:, :, 3] = 255 # opaque
    
    # 20% black
    img[0:20, 0:100] = [0, 0, 0, 255]
    
    # 30% near white
    img[20:50, 0:100] = [240, 240, 240, 255]
    
    # 10% partially transparent
    img[50:60, 0:100, 3] = 0 # transparent
    
    res = analyzer.analyze_image(img)
    
    assert res["black_pixel_ratio"] > 0
    assert res["near_black_ratio"] >= res["black_pixel_ratio"]
    assert res["alpha_empty_ratio"] > 0
    assert res["atlas_coverage_ratio"] < 1.0
    assert "dominant_color_ratio" in res
    assert "default_fill_or_flat_color_ratio" in res

def test_validator_includes_all_metrics():
    validator = AssetValidator()
    asset_data = {
        "poly_count": 100,
        "texture_integrity_status": "complete",
        "material_semantic_status": "pbr_complete",
        "texture_quality_status": "clean",
        "black_pixel_ratio": 0.1,
        "near_black_ratio": 0.2,
        "near_white_ratio": 0.3,
        "dominant_color_ratio": 0.4,
        "dominant_background_color_ratio": 0.05,
        "atlas_coverage_ratio": 0.9,
        "default_fill_or_flat_color_ratio": 0.15,
        "alpha_empty_ratio": 0.05,
        "bbox": {"x": 10, "y": 10, "z": 10},
        "ground_offset": 0.0
    }
    
    report = validator.validate("test", asset_data)
    
    assert report.black_pixel_ratio == 0.1
    assert report.near_black_ratio == 0.2
    assert report.near_white_ratio == 0.3
    assert report.dominant_color_ratio == 0.4
    assert report.dominant_background_color_ratio == 0.05
    assert report.atlas_coverage_ratio == 0.9
    assert report.default_fill_or_flat_color_ratio == 0.15
    assert report.alpha_empty_ratio == 0.05
