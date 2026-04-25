import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType
from modules.asset_cleanup_pipeline.isolation import MeshIsolator
from modules.asset_cleanup_pipeline.remesher import Remesher

@pytest.fixture
def temp_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace

def test_production_cleanup_path_auto_detects_textured_obj(temp_workspace, monkeypatch):
    """
    REQUIRED FIX 4: Regression test for production cleanup path.
    Verifies that a valid textured OBJ bundle bypasses destructive logic
    even when called with a default profile (not explicit TEXTURE_SAFE_COPY).
    """
    # 1. Setup valid textured OBJ bundle
    src_dir = temp_workspace / "source"
    src_dir.mkdir()
    
    mesh_path = src_dir / "project_textured.obj"
    mtl_path = src_dir / "project_textured.mtl"
    tex_path = src_dir / "project_textured_material_00_map_Kd.jpg"
    
    with open(mesh_path, "w") as f:
        f.write("mtllib project_textured.mtl\n")
        f.write("usemtl material_0\n")
        f.write("v 0 0 0\n")
        f.write("v 1 0 0\n")
        f.write("v 0 1 0\n")
        f.write("vt 0 0\n")
        f.write("vt 1 0\n")
        f.write("vt 0 1\n")
        f.write("f 1/1 2/2 3/3\n")
        
    with open(mtl_path, "w") as f:
        f.write("newmtl material_0\n")
        f.write("map_Kd project_textured_material_00_map_Kd.jpg\n")
        
    with open(tex_path, "w") as f:
        f.write("fake image data")
        
    # 2. Monkeypatch destructive components to fail if called
    def fail_if_called(*args, **kwargs):
        pytest.fail("Destructive cleanup path was entered but should have been bypassed!")

    monkeypatch.setattr(MeshIsolator, "isolate_product", fail_if_called)
    monkeypatch.setattr(Remesher, "process", fail_if_called)
    
    # 3. Initialize cleaner and run cleanup with DEFAULT profile
    cleaner = AssetCleaner(data_root=str(temp_workspace / "data"))
    
    metadata, stats, cleaned_path = cleaner.process_cleanup(
        job_id="prod_test_job",
        raw_mesh_path=str(mesh_path),
        profile_type=CleanupProfileType.MOBILE_DEFAULT, # NOT texture_safe_copy
        raw_texture_path=str(tex_path)
    )
    
    # 4. Assertions
    assert stats["cleanup_mode"] == "texture_safe_copy", "Should have auto-detected texture_safe_copy"
    assert Path(stats["cleaned_mesh_path"]).exists()
    assert Path(stats["cleaned_texture_path"]).exists()
    
    cleaned_mtl = Path(stats["cleaned_mesh_path"]).parent / "project_textured.mtl"
    assert cleaned_mtl.exists()
    
    # Verify metadata was generated
    assert metadata.final_polycount > 0
    assert "metadata_path" in stats
    assert Path(stats["metadata_path"]).exists()

    print("\n[SUCCESS] Production cleanup path bypassed destructive logic and used texture_safe_copy.")
