import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.operations.worker import IngestionWorker
from modules.shared_contracts.models import CaptureSession
from modules.shared_contracts.lifecycle import AssetStatus
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata

@pytest.fixture
def capture_env(tmp_path):
    # Setup directories
    data_root = tmp_path / "data"
    data_root.mkdir()
    sessions_dir = data_root / "sessions"
    sessions_dir.mkdir()
    recons_dir = data_root / "reconstructions"
    recons_dir.mkdir()
    
    # Create session
    session_id = "test_session_123"
    job_id = f"job_{session_id}"
    
    session = CaptureSession(
        session_id=session_id,
        user_id="user_1",
        operator_id="operator_1",
        product_id="prod_1",
        status=AssetStatus.RECONSTRUCTED,
        reconstruction_job_id=job_id
    )
    
    session_file = sessions_dir / f"{session_id}.json"
    with open(session_file, "w", encoding="utf-8") as f:
        f.write(session.model_dump_json())
        
    # Create reconstruction job dir and manifest
    job_dir = recons_dir / job_id
    job_dir.mkdir()
    
    manifest = {
        "schema_version": 2,
        "job_id": job_id,
        "mesh_path": str(job_dir / "meshed-poisson.ply"),
        "log_path": str(job_dir / "colmap.log"),
        "processing_time_seconds": 10.0,
        "engine_type": "colmap",
        "texturing_status": "absent"
    }
    
    manifest_path = job_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
        
    # Create fake colmap dense structure
    dense_dir = job_dir / "dense"
    dense_dir.mkdir()
    
    return data_root, session, manifest_path

@patch("modules.asset_cleanup_pipeline.cleaner.AssetCleaner.process_cleanup")
@patch("modules.reconstruction_engine.openmvs_texturer.OpenMVSTexturer.run_texturing")
@patch("trimesh.load")
def test_worker_handle_cleanup_texturing_flow(mock_trimesh_load, mock_run_texturing, mock_cleaner, capture_env):
    data_root, session, manifest_path = capture_env
    
    job_id = f"job_{session.session_id}"
    job_dir = data_root / "reconstructions" / job_id
    cleaned_dir = data_root / "cleaned" / job_id
    cleaned_dir.mkdir(parents=True)
    
    # Fake artifacts paths created by cleaner
    pre_aligned_path = str(cleaned_dir / "pre_aligned_mesh.obj")
    cleaned_mesh_path = str(cleaned_dir / "cleaned_mesh.obj")
    
    # Mock cleaner output
    fake_metadata = NormalizedMetadata(
        bbox_min={"x": -1.0, "y": -1.0, "z": 0.0},
        bbox_max={"x": 1.0, "y": 1.0, "z": 2.0},
        pivot_offset={"x": 10.5, "y": -5.2, "z": 2.0},  # Shifting values
        final_polycount=5000
    )
    fake_stats = {
        "pre_aligned_mesh_path": pre_aligned_path,
        "cleaned_mesh_path": cleaned_mesh_path,
    }
    mock_cleaner.return_value = (fake_metadata, fake_stats, cleaned_mesh_path)
    
    # Prepare fake textured output (what OpenMVSTexturer outputs)
    texturing_dir = cleaned_dir / "texturing"
    texturing_dir.mkdir()
    textured_mesh_path = texturing_dir / "textured_model.obj"
    
    # We must physically write a fake obj so worker can mathematically translate it
    with open(textured_mesh_path, "w", encoding="utf-8") as f:
        f.write("# fake obj\n")
        f.write("v 0.000 0.000 0.000\n")
        f.write("v 1.000 1.000 1.000\n")
        f.write("vt 0.5 0.5\n")
        f.write("f 1/1 2/1 3/1\n")
        
    mock_run_texturing.return_value = {
        "textured_mesh_path": str(textured_mesh_path),
        "texture_atlas_paths": [str(texturing_dir / "atlas.png")],
        "texturing_engine": "openmvs",
        "log_path": str(texturing_dir / "texturing.log")
    }
    
    # Mock trimesh returned visual properties
    mock_scene = MagicMock()
    mock_scene.visual.uv = [[0.5, 0.5]]
    mock_scene.vertices = [1, 2]
    mock_scene.faces = [1]
    mock_trimesh_load.return_value = mock_scene
    
    # EXECUTE worker stage
    worker = IngestionWorker(data_root=str(data_root))
    updated_session = worker._handle_cleanup(session)
    
    assert updated_session.status == AssetStatus.CLEANED
    
    # VERIFY run_texturing was called with pre_aligned_mesh
    mock_run_texturing.assert_called_once()
    kwargs = mock_run_texturing.call_args[1]
    assert kwargs["selected_mesh"] == pre_aligned_path
    
    # VERIFY mathematically shifted object was created
    aligned_textured_obj = cleaned_dir / "textured_aligned_mesh.obj"
    assert aligned_textured_obj.exists()
    
    # Verify the values were shifted inside the OBJ (pivot x=10.5, y=-5.2, z=2.0)
    with open(aligned_textured_obj, "r", encoding="utf-8") as f:
        content = f.read()
        assert "v 10.500000 -5.200000 2.000000" in content
        assert "v 11.500000 -4.200000 3.000000" in content
        assert "vt 0.5 0.5" in content # ensure UVs are untouched
        
    # VERIFY manifest updated accurately
    with open(manifest_path, "r", encoding="utf-8") as f:
        updated_manifest = json.load(f)
        
    assert updated_manifest["texturing_status"] == "real"
    assert "textured_aligned_mesh.obj" in updated_manifest["textured_mesh_path"]
    assert "textured_aligned_mesh.obj" in updated_manifest["mesh_path"]
    
    assert len(updated_manifest["texture_atlas_paths"]) == 1
    assert "atlas.png" in updated_manifest["texture_atlas_paths"][0]
