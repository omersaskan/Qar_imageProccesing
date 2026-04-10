import pytest
import trimesh
from PIL import Image
from modules.export_pipeline.glb_exporter import GLBExporter

@pytest.fixture
def test_dir(tmp_path):
    out = tmp_path / "pbr_tests"
    out.mkdir()
    return out

def create_pbr_obj(path, slots):
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]]
    )
    img = Image.new("RGBA", (16, 16), "red")
    
    pbr_args = {}
    if "baseColor" in slots: pbr_args["baseColorTexture"] = img
    if "normal" in slots: pbr_args["normalTexture"] = img
    if "metallicRoughness" in slots: pbr_args["metallicRoughnessTexture"] = img
    if "emissive" in slots: pbr_args["emissiveTexture"] = img
    if "occlusion" in slots: pbr_args["occlusionTexture"] = img
    
    mat = trimesh.visual.material.PBRMaterial(**pbr_args)
    mesh.visual = trimesh.visual.TextureVisuals(uv=[[0,0], [1,0], [0,1]], material=mat)
    mesh.export(str(path))

def test_inspect_diffuse_only(test_dir):
    glb_path = test_dir / "diffuse.glb"
    create_pbr_obj(glb_path, ["baseColor"])
    
    exporter = GLBExporter()
    res = exporter.inspect_exported_asset(str(glb_path))
    
    assert res["basecolor_present"] is True
    assert res["normal_present"] is False
    assert res["metallic_roughness_present"] is False
    assert res["material_semantic_status"] == "diffuse_textured"

def test_inspect_pbr_partial(test_dir):
    glb_path = test_dir / "partial.glb"
    create_pbr_obj(glb_path, ["baseColor", "normal"])
    
    exporter = GLBExporter()
    res = exporter.inspect_exported_asset(str(glb_path))
    
    assert res["basecolor_present"] is True
    assert res["normal_present"] is True
    assert res["metallic_roughness_present"] is False
    assert res["material_semantic_status"] == "pbr_partial"

def test_inspect_pbr_complete(test_dir):
    glb_path = test_dir / "complete.glb"
    create_pbr_obj(glb_path, ["baseColor", "normal", "metallicRoughness"])
    
    exporter = GLBExporter()
    res = exporter.inspect_exported_asset(str(glb_path))
    
    assert res["basecolor_present"] is True
    assert res["normal_present"] is True
    assert res["metallic_roughness_present"] is True
    assert res["material_semantic_status"] == "pbr_complete"

def test_inspect_geometry_only(test_dir):
    # Pure geometry manually
    mesh = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]], faces=[[0,1,2]])
    glb_path = test_dir / "geom.glb"
    mesh.export(str(glb_path))
    
    exporter = GLBExporter()
    res = exporter.inspect_exported_asset(str(glb_path))
    
    assert res["has_uv"] is False
    assert res["material_semantic_status"] == "geometry_only"
