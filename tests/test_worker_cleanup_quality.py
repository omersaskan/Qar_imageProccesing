import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.operations.worker import IngestionWorker
from modules.capture_workflow.session_manager import SessionManager
from modules.shared_contracts.lifecycle import AssetStatus
from modules.reconstruction_engine.output_manifest import OutputManifest, MeshMetadata

@pytest.fixture
def clean_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sessions").mkdir()
    (data_dir / "captures").mkdir()
    (data_dir / "reconstructions").mkdir()
    (data_dir / "registry").mkdir()
    
    monkeypatch.chdir(tmp_path)
    return data_dir

def test_worker_cleanup_quality_fail_marks_recapture(clean_data, monkeypatch):
    manager = SessionManager(data_root=str(clean_data))
    worker = IngestionWorker(data_root=str(clean_data))
    
    # 1. Create a session in RECONSTRUCTED status
    session = manager.create_session("sess_cleanup_fail", "prod_1", "op_1")
    manager.update_session("sess_cleanup_fail", new_status=AssetStatus.CAPTURED)
    manager.update_session("sess_cleanup_fail", new_status=AssetStatus.RECONSTRUCTED)
    
    # Setup job dir and manifest
    job_dir = clean_data / "reconstructions" / "job_sess_cleanup_fail"
    job_dir.mkdir(parents=True)
    
    # Create a real manifest file
    manifest_data = {
        "job_id": "job_sess_cleanup_fail",
        "mesh_path": str(job_dir / "mesh.ply"),
        "log_path": str(job_dir / "recon.log"),
        "processing_time_seconds": 120.0,
        "mesh_metadata": {
            "vertex_count": 1000,
            "face_count": 2000,
            "has_texture": False,
            "uv_present": False
        },
        "texturing_status": "none"
    }
    manifest_path = job_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)
    
    manager.update_session("sess_cleanup_fail", reconstruction_manifest_path=str(manifest_path))
    
    # 2. Mock AssetCleaner to return quality_fail
    mock_metadata = MagicMock()
    mock_metadata.pivot_offset = {"x":0, "y":0, "z":0}
    mock_stats = {
        "quality_status": "quality_fail",
        "quality_reason": "Low polycount",
        "final_polycount": 3000
    }
    
    monkeypatch.setattr(worker.cleaner, "process_cleanup", 
                        MagicMock(return_value=(mock_metadata, mock_stats, str(job_dir / "cleaned.obj"))))
    
    # 3. Run worker
    worker._process_pending_sessions()
    
    # 4. Verify session is now RECAPTURE_REQUIRED
    updated_session = manager.get_session("sess_cleanup_fail")
    assert updated_session.status == AssetStatus.RECAPTURE_REQUIRED
    assert "Low polycount" in updated_session.failure_reason
    assert updated_session.publish_state == "needs_recapture"
    
    # Verify cleanup_stats.json was written
    stats_path = job_dir / "cleanup_stats.json"
    assert stats_path.exists()
    with open(stats_path, "r") as f:
        saved_stats = json.load(f)
        assert saved_stats["quality_status"] == "quality_fail"
