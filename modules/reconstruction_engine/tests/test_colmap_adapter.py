import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.reconstruction_engine.adapter import COLMAPAdapter

def test_colmap_adapter_init():
    os.environ["RECON_ENGINE_PATH"] = "C:/colmap/COLMAP.bat"
    os.environ["RECON_USE_GPU"] = "false"
    os.environ["RECON_MAX_IMAGE_SIZE"] = "1000"
    
    adapter = COLMAPAdapter()
    assert adapter._engine_path == "C:/colmap/COLMAP.bat"
    assert adapter._use_gpu is False
    assert adapter._max_image_size == 1000

@patch("modules.reconstruction_engine.adapter.subprocess.Popen")
@patch("modules.reconstruction_engine.adapter.shutil.copy2")
def test_colmap_adapter_run(mock_copy, mock_popen, tmp_path):
    # Setup
    engine_path = "colmap.exe"
    adapter = COLMAPAdapter(engine_path=engine_path)
    
    input_frames = [str(tmp_path / "frame1.jpg"), str(tmp_path / "frame2.jpg")]
    for f in input_frames:
        Path(f).touch()
        
    output_dir = tmp_path / "workspace"
    output_dir.mkdir()
    
    # Mock subprocess
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = iter(["loading...", "done."])
    mock_popen.return_value = mock_process
    
    # Mock artifact existing
    mesh_file = output_dir / "meshed-poisson.ply"
    mesh_file.touch()
    
    # Run
    results = adapter.run_reconstruction(input_frames, output_dir)
    
    # Assertions
    assert results["mesh_path"] == str(mesh_file)
    assert results["log_path"] == str(output_dir / "reconstruction.log")
    
    # Verify copy was called
    assert mock_copy.call_count == 2
    
    # Verify command
    expected_cmd = [
        engine_path, "automatic_reconstructor",
        "--workspace_path", str(output_dir),
        "--image_path", str(output_dir / "images"),
        "--use_gpu", "1"
    ]
    # Re-init adapter without env to check default cmd
    adapter_default = COLMAPAdapter(engine_path=engine_path)
    # mock things again for this instance if needed, or just check the call
    
    call_args = mock_popen.call_args[0][0]
    assert call_args[0] == engine_path
    assert "automatic_reconstructor" in call_args
    assert "--workspace_path" in call_args

def test_colmap_artifact_discovery_fallback(tmp_path):
    adapter = COLMAPAdapter(engine_path="fake")
    output_dir = tmp_path / "workspace"
    output_dir.mkdir()
    
    # Create unexpected PLY
    dense_dir = output_dir / "dense" / "0"
    dense_dir.mkdir(parents=True)
    fused_ply = dense_dir / "fused.ply"
    fused_ply.touch()
    
    # We need to mock images copy too or create files
    with patch("modules.reconstruction_engine.adapter.shutil.copy2"):
        # We need to mock subprocess just to get past it
        with patch("modules.reconstruction_engine.adapter.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = iter([])
            mock_popen.return_value = mock_process
            
            results = adapter.run_reconstruction([], output_dir)
            assert results["mesh_path"] == str(fused_ply)
