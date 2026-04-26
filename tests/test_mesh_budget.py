import pytest
import os
import shutil
import trimesh
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.operations.settings import settings
from modules.utils.mesh_inspection import get_mesh_stats_cheaply

@pytest.fixture
def temp_job_dir(tmp_path):
    job_id = "test_budget_job"
    job_dir = tmp_path / "data" / "cleaned" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path, job_id

def create_fake_ply(path, face_count):
    """Creates an ASCII PLY file with a specific face count in the header."""
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {face_count * 3}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {face_count}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        # Just write one dummy vertex and face if needed, 
        # but for cheap inspection we only care about the header.
        # However, trimesh.load will fail if the body is empty.
        for i in range(face_count * 3):
            f.write(f"0.0 0.0 {i}\n")
        for i in range(face_count):
            f.write(f"3 {i*3} {i*3+1} {i*3+2}\n")

def test_cheap_inspection():
    test_ply = Path("test_cheap.ply")
    create_fake_ply(test_ply, 1000)
    try:
        stats = get_mesh_stats_cheaply(str(test_ply))
        assert stats["face_count"] == 1000
        assert stats["vertex_count"] == 3000
    finally:
        if test_ply.exists():
            os.remove(test_ply)

def test_mesh_budget_gate_trigger(temp_job_dir):
    tmp_path, job_id = temp_job_dir
    raw_mesh = tmp_path / "huge_mesh.ply"
    
    # Create a mesh that exceeds the default budget (2M) but stays under hard limit (15M)
    # For testing, we'll lower the budget in settings
    with patch.object(settings, 'recon_mesh_budget_faces', 100):
        with patch.object(settings, 'recon_pre_cleanup_target_faces', 50):
            create_fake_ply(raw_mesh, 150)
            
            cleaner = AssetCleaner(data_root=str(tmp_path / "data"))
            
            # Mock the isolator and other parts to avoid actual heavy processing
            cleaner.isolator.isolate_product = MagicMock(return_value=(trimesh.creation.box(), {"object_isolation_status": "success"}))
            cleaner.remesher.process = MagicMock(return_value={"post_decimation_face_count": 10})
            cleaner.alignment.align_to_ground = MagicMock(return_value=(None, {"x":0,"y":0,"z":0}))
            cleaner.bbox_extractor.extract = MagicMock(return_value=({"x":0,"y":0,"z":0}, {"x":1,"y":1,"z":1}))
            
            metadata, stats, mesh_path = cleaner.process_cleanup(job_id, str(raw_mesh))
            
            assert stats["oversized_raw_mesh"] is True
            assert stats["raw_mesh_faces"] == 150
            # Ensure it went through isolation and other steps
            assert cleaner.isolator.isolate_product.called

def test_mesh_hard_limit_fail(temp_job_dir):
    tmp_path, job_id = temp_job_dir
    raw_mesh = tmp_path / "too_huge_mesh.ply"
    
    with patch.object(settings, 'recon_mesh_hard_limit_faces', 500):
        create_fake_ply(raw_mesh, 1000)
        
        cleaner = AssetCleaner(data_root=str(tmp_path / "data"))
        metadata, stats, mesh_path = cleaner.process_cleanup(job_id, str(raw_mesh))
        
        assert metadata is None
        assert stats["status"] == "failed_oversized_mesh"
        assert stats["raw_faces"] == 1000
        assert mesh_path == ""

if __name__ == "__main__":
    # Quick manual run
    test_cheap_inspection()
    print("Cheap inspection test passed")
