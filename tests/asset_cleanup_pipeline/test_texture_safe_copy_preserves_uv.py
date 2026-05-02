import os
import pytest
import trimesh
import numpy as np
from pathlib import Path
from PIL import Image
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def textured_obj(temp_dir):
    mesh_path = temp_dir / "input.obj"
    mtl_path = temp_dir / "material.mtl"
    tex_path = temp_dir / "texture.png"
    
    # Create a proper box mesh with enough faces (>50)
    import trimesh
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    while len(mesh.faces) < 60:
        mesh = mesh.subdivide()
    
    # Export to OBJ with UVs
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.random.rand(len(mesh.vertices), 2))
    mesh.export(str(mesh_path))
    
    with open(mtl_path, "w") as f:
        f.write("newmtl material_0\nmap_Kd texture.png\n")
        
    img = Image.new('RGB', (100, 100), color=(0, 255, 0))
    img.save(str(tex_path))
    
    return str(mesh_path), str(tex_path)

def test_texture_safe_copy_preserves_uv_and_mtl(textured_obj, temp_dir):
    mesh_path, tex_path = textured_obj
    cleaner = AssetCleaner(data_root=str(temp_dir))
    
    result_metadata, stats, cleaned_path = cleaner.process_cleanup(
        job_id="test_safe_copy",
        raw_mesh_path=mesh_path,
        profile_type=CleanupProfileType.TEXTURE_SAFE_COPY,
        raw_texture_path=tex_path,
        cameras=[{"name": "test.jpg", "P": np.eye(3, 4)}],
        masks={"test.jpg": np.ones((100, 100), dtype=np.uint8) * 255}
    )
    
    assert stats["cleanup_mode"] == "texture_safe_copy"
    assert stats["uv_preserved"] is True
    assert stats["delivery_ready"] is True
    
    # Verify the cleaned OBJ still has vt lines and usemtl
    vt_found = False
    usemtl_found = False
    with open(cleaned_path, "r") as f:
        for line in f:
            if line.startswith("vt "): vt_found = True
            if line.startswith("usemtl "): usemtl_found = True
            
    assert vt_found is True
    assert usemtl_found is True
    
    # Verify MTL exists and points to texture
    cleaned_dir = Path(cleaned_path).parent
    mtl_path = cleaned_dir / "material.mtl"
    assert mtl_path.exists()
    
    with open(mtl_path, "r") as f:
        content = f.read()
        assert "map_Kd" in content
        assert Path(tex_path).name in content

def test_texture_safe_copy_alignment_preserves_uv(textured_obj, temp_dir):
    mesh_path, tex_path = textured_obj
    cleaner = AssetCleaner(data_root=str(temp_dir))
    
    # Run the internal safe align
    out_path = temp_dir / "aligned.obj"
    pivot, bmin, bmax = cleaner._safe_align_obj(Path(mesh_path), out_path)
    
    # Verify aligned OBJ
    vt_count = 0
    with open(out_path, "r") as f:
        for line in f:
            if line.startswith("vt "): vt_count += 1
            
    assert vt_count > 0
