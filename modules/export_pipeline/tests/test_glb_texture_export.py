import pytest
import numpy as np
import trimesh
from pathlib import Path
from PIL import Image

from modules.export_pipeline.glb_exporter import GLBExporter

@pytest.fixture
def test_dir(tmp_path):
    out = tmp_path / "export_tests"
    out.mkdir()
    return out

def create_geometry_only_obj(path):
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]]
    )
    mesh.export(str(path))

def create_uv_only_obj(path):
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]]
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=[[0,0], [1,0], [0,1]])
    mesh.export(str(path))

@pytest.fixture
def dummy_texture(test_dir):
    tex_path = test_dir / "dummy_tex.png"
    img = Image.new("RGBA", (16, 16), "red")
    img.save(tex_path)
    return str(tex_path)

def test_export_geometry_only(test_dir):
    obj_path = test_dir / "geom.obj"
    glb_path = test_dir / "geom.glb"
    create_geometry_only_obj(obj_path)
    
    exporter = GLBExporter()
    res = exporter.export_to_glb(str(obj_path), None, str(glb_path))
    
    assert res["has_uv"] is False
    assert res["has_material"] is True
    assert res["has_embedded_texture"] is False
    assert res["material_semantic_status"] == "geometry_only"
    assert res["texture_integrity_status"] == "missing"
    assert res["texture_applied_successfully"] is False
    
    inspect_res = exporter.inspect_exported_asset(str(glb_path))
    assert inspect_res["has_uv"] is False
    assert inspect_res["has_material"] is True  # fallback material is applied!
    assert inspect_res["has_embedded_texture"] is False
    assert inspect_res["texture_integrity_status"] == "missing" # no UV and no texture = missing
    assert inspect_res["material_semantic_status"] == "geometry_only"



def test_export_uv_only(test_dir):
    obj_path = test_dir / "uv.obj"
    glb_path = test_dir / "uv.glb"
    create_uv_only_obj(obj_path)
    
    exporter = GLBExporter()
    res = exporter.export_to_glb(str(obj_path), None, str(glb_path))
    
    assert res["has_uv"] is True
    
    inspect_res = exporter.inspect_exported_asset(str(glb_path))
    assert inspect_res["has_uv"] is True
    assert inspect_res["has_embedded_texture"] is False
    assert inspect_res["texture_integrity_status"] == "degraded"
    assert inspect_res["material_semantic_status"] == "uv_only"


def test_export_textured_complete(test_dir, dummy_texture):
    obj_path = test_dir / "tex.obj"
    glb_path = test_dir / "tex.glb"
    create_uv_only_obj(obj_path)
    
    exporter = GLBExporter()
    res = exporter.export_to_glb(str(obj_path), dummy_texture, str(glb_path))
    
    assert res["has_uv"] is True
    assert res["texture_applied_successfully"] is True
    
    inspect_res = exporter.inspect_exported_asset(str(glb_path))
    assert inspect_res["has_uv"] is True
    assert inspect_res["has_material"] is True
    assert inspect_res["has_embedded_texture"] is True
    assert inspect_res["texture_integrity_status"] == "complete"
    assert inspect_res["material_semantic_status"] == "diffuse_textured"
    assert inspect_res["texture_count"] == 1
    assert inspect_res["material_count"] == 1

