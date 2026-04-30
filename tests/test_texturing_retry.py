import pytest
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.reconstruction_engine.openmvs_texturer import OpenMVSTexturer

@patch("subprocess.Popen")
def test_openmvs_timeout_handling(mock_popen, tmp_path):
    # Setup mock process
    mock_process = MagicMock()
    mock_process.poll.return_value = None  # Still running
    mock_process.stdout.readline.return_value = "Processing...\n"
    mock_process.communicate.return_value = ("Final output after kill", None)
    mock_popen.return_value = mock_process
    
    texturer = OpenMVSTexturer(bin_dir=str(tmp_path))
    
    log_path = tmp_path / "texturing.log"
    with open(log_path, "w") as log_file:
        # We expect this to raise RuntimeError because of timeout
        with pytest.raises(RuntimeError) as exc:
            texturer._run_command(["test_cmd"], tmp_path, log_file, timeout=1)
            
    assert "timed out" in str(exc.value)
    mock_process.kill.assert_called()
    
    # Check if log contains the marker and drained output
    with open(log_path, "r") as f:
        content = f.read()
        assert "!!! ERROR: TIMEOUT EXCEEDED !!!" in content
        assert "Final output after kill" in content

@patch("subprocess.Popen")
def test_openmvs_success(mock_popen, tmp_path):
    # Setup mock process for success
    mock_process = MagicMock()
    mock_process.poll.side_effect = [None, 0] # Running then finished
    mock_process.stdout.readline.side_effect = ["Output line 1\n", ""]
    mock_process.returncode = 0
    mock_popen.return_value = mock_process
    
    texturer = OpenMVSTexturer(bin_dir=str(tmp_path))
    
    log_path = tmp_path / "texturing_success.log"
    with open(log_path, "w") as log_file:
        texturer._run_command(["test_cmd"], tmp_path, log_file, timeout=10)
        
    mock_process.kill.assert_not_called()
    with open(log_path, "r") as f:
        content = f.read()
        assert "Output line 1" in content
        assert "!!! ERROR: TIMEOUT EXCEEDED !!!" not in content
