import pytest
import trimesh
import numpy as np
import os
import struct
import json
from pathlib import Path
from PIL import Image
from modules.export_pipeline.glb_exporter import GLBExporter, inspect_glb_primitive_attributes
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

def create_glb_without_normals(source_glb, target_glb):
    with open(source_glb, "rb") as f:
        f.read(12)  # skip header
        chunk_len = struct.unpack("<I", f.read(4))[0]
        chunk_type = f.read(4)
        json_data = f.read(chunk_len).decode("utf-8")
        rest = f.read()

    gltf = json.loads(json_data)
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "NORMAL" in prim.get("attributes", {}):
                del prim["attributes"]["NORMAL"]

    new_json = json.dumps(gltf).encode("utf-8")
    while len(new_json) % 4 != 0:
        new_json += b" "

    with open(target_glb, "wb") as f:
        f.write(b"glTF")
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<I", 12 + 8 + len(new_json) + len(rest)))
        f.write(struct.pack("<I", len(new_json)))
        f.write(b"JSON")
        f.write(new_json)
        f.write(rest)

def test_strict_glb_inspector(test_dir, dummy_texture):
    obj_path = test_dir / "base.obj"
    glb_path = test_dir / "base.glb"
    no_norm_glb = test_dir / "no_norm.glb"
    create_uv_only_obj(obj_path)
    
    exporter = GLBExporter()
    exporter.export_to_glb(str(obj_path), dummy_texture, str(glb_path))
    
    # 1. Normal GLB should have all accessors
    res = inspect_glb_primitive_attributes(str(glb_path))
    assert res["has_position_accessor"] is True
    assert res["has_normal_accessor"] is True
    assert res["has_texcoord_0_accessor"] is True
    
    # 2. Manipulated GLB should lack NORMAL
    create_glb_without_normals(str(glb_path), str(no_norm_glb))
    res_bad = inspect_glb_primitive_attributes(str(no_norm_glb))
    assert res_bad["has_position_accessor"] is True
    assert res_bad["has_normal_accessor"] is False
    assert res_bad["has_texcoord_0_accessor"] is True

def test_glb_normals_and_accessors_fix(test_dir, dummy_texture):
    obj_path = test_dir / "test.obj"
    glb_path = test_dir / "test.glb"
    create_uv_only_obj(obj_path)
    
    exporter = GLBExporter()
    res = exporter.export_to_glb(str(obj_path), dummy_texture, str(glb_path))
    
    assert res["has_position_accessor"] is True
    assert res["has_normal_accessor"] is True
    assert res["has_texcoord_0_accessor"] is True
    
    # Reload and check strictly via JSON
    strict = inspect_glb_primitive_attributes(str(glb_path))
    assert strict["has_normal_accessor"] is True

def test_missing_normal_fails_textured_export(test_dir, dummy_texture, monkeypatch):
    obj_path = test_dir / "test_fail.obj"
    glb_path = test_dir / "test_fail.glb"
    create_uv_only_obj(obj_path)
    
    exporter = GLBExporter()
    
    # Test 1: High-level rejection logic
    # We create a bad GLB and verify that inspection catches it.
    exporter.export_to_glb(str(obj_path), dummy_texture, str(glb_path))
    create_glb_without_normals(str(glb_path), str(glb_path))
    
    res = exporter.inspect_exported_asset(str(glb_path))
    assert res["has_normal_accessor"] is False
    
    # Test 2: Verify the gate raises ValueError if Normal is missing on a textured asset
    # To trigger the gate inside export_to_glb without deep monkeypatching, 
    # we can mock the internal trimesh export to produce a bad file.
    
    # But since we've already verified inspect_exported_asset is strict, 
    # and the code in export_to_glb is:
    #   inspection_result = self.inspect_exported_asset(output_path)
    #   if actual_texture_success and not inspection_result["has_normal_accessor"]:
    #       raise ValueError(...)
    
    # We'll do a focused test for this gate by mocking inspect_exported_asset.
    # This is a unit test for the gate itself.
    orig_inspect = exporter.inspect_exported_asset
    def mock_inspect(path):
        real_res = orig_inspect(path)
        real_res["has_normal_accessor"] = False
        return real_res
    
    with pytest.raises(ValueError, match="Textured GLB must have NORMAL accessor"):
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(exporter, "inspect_exported_asset", mock_inspect)
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
        "ground_offset": 0.0,
        "has_position_accessor": True,
        "has_normal_accessor": True,
        "has_texcoord_0_accessor": True,
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
        "ground_offset": 0.0,
        "has_position_accessor": True,
        "has_normal_accessor": True,
        "has_texcoord_0_accessor": True,
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
        "ground_offset": 0.0,
        "has_position_accessor": True,
        "has_normal_accessor": True,
        "has_texcoord_0_accessor": True,
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
        "ground_offset": 0.0,
        "has_position_accessor": True,
        "has_normal_accessor": True,
        "has_texcoord_0_accessor": True,
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
        "texture_quality_status": "clean",
        "has_position_accessor": True,
        "has_normal_accessor": True,
        "has_texcoord_0_accessor": True,
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
