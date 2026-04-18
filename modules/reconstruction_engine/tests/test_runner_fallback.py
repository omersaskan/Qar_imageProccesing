from pathlib import Path
from unittest.mock import MagicMock

import pytest

from modules.shared_contracts.models import ReconstructionJobDraft
from modules.reconstruction_engine.job_manager import JobManager
from modules.reconstruction_engine.runner import ReconstructionRunner
from modules.reconstruction_engine.failures import RuntimeReconstructionError
from modules.reconstruction_engine.adapter import COLMAPAdapter, OpenMVSAdapter

def test_runner_fallback_engine_write(tmp_path, monkeypatch):
    """
    Test that if OpenMVS fails and COLMAP fallback wins, the final
    manifest.engine_type correctly reflects the colmap fallback engine instead
    of the failed primary engine.
    """
    manager = JobManager(data_root=str(tmp_path))
    
    # Create some mock input frames
    input_frames = []
    for i in range(3):
        frame_path = tmp_path / f"frame_{i}.jpg"
        frame_path.write_bytes(b"dummy image bytes")  # Runner does a validation that needs exist() 
        input_frames.append(str(frame_path))
        
    draft = ReconstructionJobDraft(
        job_id="RJ_FALLBACK_TEST",
        capture_session_id="S1",
        input_frames=input_frames,
        product_id="P1",
    )
    job = manager.create_job(draft)
    
    runner = ReconstructionRunner()
    
    # We pretend OpenMVS is selected as the primary engine
    runner.adapter = OpenMVSAdapter()
    
    # We must patch the validation of input frames since they are dummy bytes
    monkeypatch.setattr(runner, "_validate_input_frames", lambda x: x)
    
    # Mock OpenMVS run_reconstruction to throw an error
    def mock_openmvs_run(*args, **kwargs):
        raise RuntimeReconstructionError("OpenMVS crashed")
        
    monkeypatch.setattr(runner.adapter, "run_reconstruction", mock_openmvs_run)
    
    # Mock COLMAP run_reconstruction to succeed
    def mock_colmap_run(*args, **kwargs):
        # Must return valid-looking result dict
        output_dir = Path(kwargs.get("output_dir", args[1]))
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = output_dir / "mesh.obj"
        texture_path = output_dir / "mesh.png"
        log_path = output_dir / "colmap.log"
        mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        texture_path.write_text("dummy_texture")
        log_path.write_text("OK")
        
        return {
            "registered_images": 3,
            "sparse_points": 100,
            "dense_points_fused": 1000,
            "mesher_used": "poisson",
            "mesh_path": str(mesh_path),
            "texture_path": str(texture_path),
            "log_path": str(log_path),
        }
        
    monkeypatch.setattr(runner.colmap_adapter, "run_reconstruction", mock_colmap_run)
    monkeypatch.setattr(runner, "_validate_mesh_artifact", lambda x: (3, 1))
    
    manifest = runner.run(job)
    
    assert manifest.job_id == "RJ_FALLBACK_TEST"
    assert manifest.engine_type == "colmap (fallback)"
