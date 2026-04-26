import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import trimesh

from modules.operations.worker import IngestionWorker
from modules.shared_contracts.lifecycle import AssetStatus
from modules.shared_contracts.models import CaptureSession, ReconstructionJob
from modules.reconstruction_engine.runner import ReconstructionRunner
from modules.reconstruction_engine.adapter import COLMAPAdapter
from modules.operations.settings import settings

@pytest.fixture
def mock_worker():
    with patch("modules.operations.worker.AssetRegistry"), \
         patch("modules.operations.worker.GuidanceAggregator"), \
         patch("modules.operations.worker.RetentionService"), \
         patch("modules.operations.worker.TexturingService"), \
         patch("modules.operations.worker.GLBExporter"), \
         patch("modules.operations.worker.AssetValidator"):
        w = IngestionWorker(data_root="data_test_retry")
        w.session_manager = MagicMock()
        return w

def test_processing_budget_exceeded_dispatch(mock_worker):
    """Verify that PROCESSING_BUDGET_EXCEEDED is dispatched to retry handler."""
    session = CaptureSession(
        session_id="sess_retry", 
        product_id="p1", 
        operator_id="o1", 
        status=AssetStatus.PROCESSING_BUDGET_EXCEEDED
    )
    
    with patch.object(mock_worker, "_handle_budget_exceeded_retry") as mock_retry:
        # Simulate worker finding the session
        mock_worker.session_manager.sessions_dir.glob.return_value = [Path("data_test_retry/sessions/sess_retry.json")]
        
        # We need to mock open() to return our session
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value=session.model_dump()):
                with patch("modules.operations.worker.CaptureSession.model_validate", return_value=session):
                    # Mock lock and timeout
                    mock_worker._check_session_timeout = MagicMock(return_value=False)
                    mock_worker._process_pending_sessions()
                    
        assert mock_retry.called

def test_cleanup_failure_routing_to_budget_exceeded(mock_worker):
    """Verify that failed_memory_limit routes to PROCESSING_BUDGET_EXCEEDED."""
    session = CaptureSession(
        session_id="sess_oom", 
        product_id="p1", 
        operator_id="o1", 
        status=AssetStatus.RECONSTRUCTED
    )
    
    with patch.object(mock_worker.cleaner, "process_cleanup") as mock_cleanup, \
         patch.object(mock_worker, "_persist_session") as mock_persist:
        
        mock_cleanup.return_value = (
            None, 
            {
                "status": "failed_memory_limit",
                "retryable_from_fused_ply": True,
                "reason": "OOM in pre-decimation"
            },
            ""
        )
        mock_worker._load_manifest = MagicMock()
        mock_worker._handle_cleanup(session)
        assert mock_persist.called
        assert mock_persist.call_args[1]["new_status"] == AssetStatus.PROCESSING_BUDGET_EXCEEDED

def test_score_attempt_optimisation(tmp_path):
    """Verify that _score_attempt avoids trimesh.load on large meshes."""
    runner = ReconstructionRunner()
    mesh_path = tmp_path / "huge_mesh.ply"
    
    # Create a "large" PLY by writing a lot of faces
    # Settings.max_faces_python_decimation is 1.5M by default
    with open(mesh_path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 2000000\nproperty list uchar int vertex_indices\nend_header\n")
        # We don't actually need the vertices/faces for get_mesh_stats_cheaply
        
    results = {"mesh_path": str(mesh_path)}
    
    with patch("trimesh.load") as mock_load:
        runner._score_attempt(results)
        # Should NOT call trimesh.load because face count (2M) > threshold (1.5M)
        assert not mock_load.called

@patch("modules.reconstruction_engine.adapter.COLMAPAdapter.poisson_remesh_only")
def test_remesh_retry_from_fused_ply(mock_remesh, tmp_path):
    """Verify remesh_retry works from existing fused.ply and logs settings."""
    runner = ReconstructionRunner()
    job_dir = tmp_path / "job_retry"
    job_dir.mkdir()
    
    attempt_dir = job_dir / "attempt_0_default"
    attempt_dir.mkdir()
    
    dense_dir = attempt_dir / "dense"
    dense_dir.mkdir()
    
    fused_ply = dense_dir / "fused.ply"
    fused_ply.write_text("dummy points")
    
    # Create dummy output mesh so _validate_mesh_artifact passes
    output_mesh = dense_dir / "meshed-poisson.ply"
    output_mesh.write_text("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n")
    
    log_path = attempt_dir / "reconstruction.log"
    log_path.write_text("initial log\n")
    
    job = ReconstructionJob(
        job_id="job_retry",
        capture_session_id="sess_retry",
        product_id="p1",
        input_frames=[],
        job_dir=str(job_dir)
    )
    
    mock_remesh.return_value = "poisson"
    
    with patch("modules.reconstruction_engine.runner.atomic_write_json") as mock_write:
        runner.remesh_retry(job, depth=8, trim=8)
        
        assert mock_remesh.called
        assert mock_remesh.call_args[0][0] == attempt_dir # First arg is output_dir (Path)
        
        # Check log file
        log_content = log_path.read_text()
        assert "PROCESSING BUDGET EXCEEDED" in log_content
        assert "depth=8, trim=8" in log_content

def test_colmap_adapter_poisson_command_generation(tmp_path):
    """Verify that poisson_remesh_only generates the correct COLMAP command."""
    adapter = COLMAPAdapter()
    output_dir = tmp_path / "recon"
    dense_dir = output_dir / "dense"
    dense_dir.mkdir(parents=True)
    fused_ply = dense_dir / "fused.ply"
    fused_ply.write_text("dummy")
    
    # Mock _run_command to capture the command
    captured_cmd = []
    def mock_run_command(cmd, cwd, log, timeout=None):
        captured_cmd.append(cmd)
    
    adapter._run_command = mock_run_command
    # Mock _is_valid_mesh_candidate to return True
    adapter._is_valid_mesh_candidate = MagicMock(return_value=True)
    
    log_file = MagicMock()
    adapter.poisson_remesh_only(output_dir, log_file, depth=9, trim=8)
    
    assert len(captured_cmd) == 1
    cmd = captured_cmd[0]
    
    # Check flags
    # Use find to check if substrings exist in the command list
    assert any("--input_path" in part for part in cmd)
    assert any(str(fused_ply) in part for part in cmd)
    assert any("--output_path" in part for part in cmd)
    assert any(str(dense_dir / "meshed-poisson.ply") in part for part in cmd)
    assert any("--PoissonMeshing.depth" in part for part in cmd)
    
    # Find depth value
    for i, part in enumerate(cmd):
        if "--PoissonMeshing.depth" in part:
            assert cmd[i+1] == "9"
        if "--PoissonMeshing.trim" in part:
            assert cmd[i+1] == "8"

if __name__ == "__main__":
    pytest.main([__file__])
