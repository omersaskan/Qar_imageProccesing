import os
import pytest
from pathlib import Path
import shutil
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType

@pytest.fixture
def temp_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace

def test_texture_safe_copy_success(temp_workspace):
    # Setup source files
    src_dir = temp_workspace / "source"
    src_dir.mkdir()
    
    mesh_path = src_dir / "input.obj"
    mtl_path = src_dir / "material.mtl"
    tex_path = src_dir / "texture.jpg"
    
    with open(mesh_path, "w") as f:
        f.write("mtllib material.mtl\n")
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
        
    with open(tex_path, "w") as f:
        f.write("fake image data")
        
    cleaner = AssetCleaner(data_root=str(temp_workspace / "data"))
    
    metadata, stats, cleaned_path = cleaner.process_cleanup(
        job_id="test_job",
        raw_mesh_path=str(mesh_path),
        profile_type=CleanupProfileType.TEXTURE_SAFE_COPY,
        raw_texture_path=str(tex_path)
    )
    
    assert stats["cleanup_mode"] == "texture_safe_copy"
    assert stats["uv_preserved"] is True
    assert stats["material_preserved"] is True
    assert Path(stats["cleaned_mesh_path"]).exists()
    assert Path(stats["cleaned_texture_path"]).exists()
    assert Path(stats["cleaned_mesh_path"]).name == "cleaned_mesh.obj"
    
    # Check MTL content
    mtl_out = Path(stats["cleaned_mesh_path"]).parent / "material.mtl"
    assert mtl_out.exists()
    with open(mtl_out, "r") as f:
        content = f.read()
        assert "map_Kd texture.jpg" in content
        assert "Ka 1.000000 1.000000 1.000000" in content

def test_texture_safe_copy_fails_missing_vt(temp_workspace):
    src_dir = temp_workspace / "source"
    src_dir.mkdir()
    
    mesh_path = src_dir / "input.obj"
    mtl_path = src_dir / "material.mtl"
    tex_path = src_dir / "texture.jpg"
    
    # No vt lines
    with open(mesh_path, "w") as f:
        f.write("mtllib material.mtl\n")
        f.write("v 0 0 0\n")
        f.write("v 1 0 0\n")
        f.write("v 0 1 0\n")
        f.write("f 1 2 3\n")
        
    with open(mtl_path, "w") as f:
        f.write("newmtl material_0\n")
        f.write("map_Kd texture.jpg\n")
        
    with open(tex_path, "w") as f:
        f.write("fake image data")
        
    cleaner = AssetCleaner(data_root=str(temp_workspace / "data"))
    
    with pytest.raises(ValueError, match="OBJ missing 'vt' lines"):
        cleaner.process_cleanup(
            job_id="test_job_fail",
            raw_mesh_path=str(mesh_path),
            profile_type=CleanupProfileType.TEXTURE_SAFE_COPY,
            raw_texture_path=str(tex_path)
        )
