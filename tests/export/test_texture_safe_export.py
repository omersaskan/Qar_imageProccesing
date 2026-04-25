import os
import pytest
from pathlib import Path
from PIL import Image
import trimesh
from modules.export_pipeline.glb_exporter import GLBExporter

@pytest.fixture
def temp_export_dir(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    return export_dir

def test_texture_safe_export_normal(temp_export_dir):
    # Create a simple textured OBJ
    obj_path = temp_export_dir / "test.obj"
    mtl_path = temp_export_dir / "test.mtl"
    tex_path = temp_export_dir / "texture.jpg"
    
    with open(obj_path, "w") as f:
        f.write("mtllib test.mtl\n")
        f.write("v 0 0 0\n")
        f.write("v 1 0 0\n")
        f.write("v 0 1 0\n")
        f.write("vt 0 0\n")
        f.write("vt 1 0\n")
        f.write("vt 0 1\n")
        f.write("f 1/1 2/2 3/3\n")
        
    with open(mtl_path, "w") as f:
        f.write("newmtl material_0\n")
        f.write("map_Kd texture.jpg\n")
        
    # Create real image
    img = Image.new("RGB", (64, 64), color="red")
    img.save(tex_path)
    
    exporter = GLBExporter()
    glb_path = temp_export_dir / "output.glb"
    
    result = exporter.export(
        mesh_path=str(obj_path),
        output_path=str(glb_path),
        texture_path=str(tex_path)
    )
    
    assert result["texture_applied_successfully"] is True
    assert result["has_uv"] is True
    assert result["material_semantic_status"] in {"diffuse_textured", "pbr_partial", "pbr_complete"}
    assert glb_path.exists()
    
    # Reload and check
    scene = trimesh.load(str(glb_path), force="scene")
    has_texture = False
    for geom in scene.geometry.values():
        if hasattr(geom.visual, "material") and hasattr(geom.visual.material, "baseColorTexture"):
            if geom.visual.material.baseColorTexture is not None:
                has_texture = True
                break
    assert has_texture is True

def test_texture_safe_export_black_mtl_prevention(temp_export_dir):
    # Create an OBJ with a black MTL
    obj_path = temp_export_dir / "black.obj"
    mtl_path = temp_export_dir / "black.mtl"
    tex_path = temp_export_dir / "white_texture.jpg"
    
    with open(obj_path, "w") as f:
        f.write("mtllib black.mtl\n")
        f.write("v 0 0 0\n")
        f.write("v 1 0 0\n")
        f.write("v 0 1 0\n")
        f.write("vt 0 0\n")
        f.write("vt 1 0\n")
        f.write("vt 0 1\n")
        f.write("f 1/1 2/2 3/3\n")
        
    with open(mtl_path, "w") as f:
        f.write("newmtl material_0\n")
        f.write("Kd 0.000000 0.000000 0.000000\n") # Black
        f.write("map_Kd white_texture.jpg\n")
        
    img = Image.new("RGB", (64, 64), color="white")
    img.save(tex_path)
    
    exporter = GLBExporter()
    glb_path = temp_export_dir / "output_black_prevented.glb"
    
    result = exporter.export(
        mesh_path=str(obj_path),
        output_path=str(glb_path),
        texture_path=str(tex_path)
    )
    
    assert result["texture_applied_successfully"] is True
    assert result["material_semantic_status"] != "geometry_only"
    
    # Reload and verify material is not black
    scene = trimesh.load(str(glb_path), force="scene")
    for geom in scene.geometry.values():
        if hasattr(geom.visual, "material"):
            mat = geom.visual.material
            if hasattr(mat, "baseColorFactor"):
                # Should be [1.0, 1.0, 1.0, 1.0] or at least not black
                assert any(c > 0.5 for c in mat.baseColorFactor[:3])
