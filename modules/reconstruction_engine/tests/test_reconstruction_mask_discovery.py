import pytest
from pathlib import Path
from unittest.mock import patch
from modules.reconstruction_engine.adapter import COLMAPAdapter

def test_prepare_workspace_resolution_modes(tmp_path):
    adapter = COLMAPAdapter(engine_path="fake")
    input_frames = []
    
    # 1. Stem-based frame
    f1 = tmp_path / "f1.jpg"
    f1.touch()
    (tmp_path / "masks").mkdir(exist_ok=True)
    (tmp_path / "masks" / "f1.png").touch()
    input_frames.append(str(f1))
    
    # 2. Legacy-based frame
    f2 = tmp_path / "f2.jpg"
    f2.touch()
    (tmp_path / "masks" / "f2.jpg.png").touch()
    input_frames.append(str(f2))
    
    # 3. Missing mask frame
    f3 = tmp_path / "f3.jpg"
    f3.touch()
    input_frames.append(str(f3))

    output_dir = tmp_path / "workspace"
    output_dir.mkdir()

    with patch.object(adapter, "_frame_is_usable", return_value=True):
        with patch.object(adapter, "_mask_is_usable", return_value=True):
            prep = adapter._prepare_workspace(input_frames, output_dir)

    assert prep["accepted_frames"] == 2
    assert prep["rejected_missing_mask"] == 1
    
    counts = prep["match_mode_counts"]
    assert counts["stem"] == 1
    assert counts["legacy"] == 1
    assert counts["none"] == 1

def test_insufficient_input_error_message(tmp_path):
    from modules.reconstruction_engine.failures import InsufficientInputError
    adapter = COLMAPAdapter(engine_path="fake")
    
    # Create one frame with no mask
    f1 = tmp_path / "f1.jpg"
    f1.touch()
    (tmp_path / "masks").mkdir(exist_ok=True)
    
    output_dir = tmp_path / "workspace"
    output_dir.mkdir()
    
    with patch.object(adapter, "_frame_is_usable", return_value=True):
        with pytest.raises(InsufficientInputError) as excinfo:
            adapter.run_reconstruction([str(f1)], output_dir)
    
    # Check that error message contains the expected counts
    error_msg = str(excinfo.value)
    assert "accepted=0" in error_msg
    assert "missing_mask=1" in error_msg
    assert "modes(stem=0, legacy=0, none=1)" in error_msg
