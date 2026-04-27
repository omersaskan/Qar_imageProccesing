import os
import pytest
from pathlib import Path
from PIL import Image
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def valid_bundle(temp_dir):
    mesh_path = temp_dir / "mesh.obj"
    mtl_path = temp_dir / "material.mtl"
    tex_path = temp_dir / "texture.png"
    
    with open(mesh_path, "w") as f:
        f.write("mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n")
    
    with open(mtl_path, "w") as f:
        f.write("newmtl material_0\nmap_Kd texture.png\n")
        
    img = Image.new('RGB', (100, 100), color=(255, 0, 0))
    img.save(str(tex_path))
    
    return str(mesh_path), str(tex_path)

def test_valid_bundle_check(valid_bundle):
    mesh_path, tex_path = valid_bundle
    cleaner = AssetCleaner()
    is_valid, msg, resolved_tex = cleaner._is_valid_textured_obj_bundle(mesh_path)
    
    assert is_valid is True
    assert "Valid" in msg
    assert Path(resolved_tex).name == "texture.png"

def test_missing_mtl_fail(valid_bundle):
    mesh_path, tex_path = valid_bundle
    os.remove(Path(mesh_path).parent / "material.mtl")
    
    cleaner = AssetCleaner()
    is_valid, msg, _ = cleaner._is_valid_textured_obj_bundle(mesh_path)
    
    assert is_valid is False
    assert "MTL file missing" in msg

def test_missing_texture_fail(valid_bundle):
    mesh_path, tex_path = valid_bundle
    os.remove(tex_path)
    
    cleaner = AssetCleaner()
    is_valid, msg, _ = cleaner._is_valid_textured_obj_bundle(mesh_path)
    
    assert is_valid is False
    assert "Texture file missing" in msg

def test_missing_uvs_fail(temp_dir):
    mesh_path = temp_dir / "no_uv.obj"
    with open(mesh_path, "w") as f:
        f.write("mtllib material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    
    cleaner = AssetCleaner()
    is_valid, msg, _ = cleaner._is_valid_textured_obj_bundle(str(mesh_path))
    
    assert is_valid is False
    assert "missing 'vt' lines" in msg
