import pytest
from pathlib import Path
import shutil
import os
import cv2
import numpy as np
from unittest.mock import MagicMock, patch

from modules.reconstruction_engine.openmvs_texturer import OpenMVSTexturer
from modules.reconstruction_engine.failures import TexturingFailed

@pytest.fixture
def temp_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    
    images_dir = workspace / "images"
    images_dir.mkdir()
    
    # Create 3 dummy images
    for i in range(3):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"frame_{i:04d}.jpg"), img)
        
    dense_dir = workspace / "dense"
    dense_dir.mkdir()
    (dense_dir / "images").mkdir()
    for i in range(3):
        shutil.copy2(images_dir / f"frame_{i:04d}.jpg", dense_dir / "images" / f"frame_{i:04d}.jpg")
        
    output_dir = workspace / "output"
    output_dir.mkdir()
    
    return workspace, dense_dir, output_dir

def test_create_compatible_image_folder(temp_workspace):
    workspace, dense_dir, output_dir = temp_workspace
    texturer = OpenMVSTexturer(bin_dir=str(workspace))
    
    original_images_dir = dense_dir / "images"
    target_dir = output_dir / "compatible_images"
    selected_names = ["frame_0000.jpg"]
    
    texturer._create_compatible_image_folder(
        original_images_dir=original_images_dir,
        target_dir=target_dir,
        selected_names=selected_names,
        log_file=MagicMock()
    )
    
    # Check all filenames exist
    for i in range(3):
        assert (target_dir / f"frame_{i:04d}.jpg").exists()
        
    # Check frame_0000.jpg is original (black)
    img0 = cv2.imread(str(target_dir / "frame_0000.jpg"))
    assert np.all(img0 == 0)
    
    # Check frame_0001.jpg is neutralized (cream)
    img1 = cv2.imread(str(target_dir / "frame_0001.jpg"))
    # Allow small tolerance for JPEG compression
    mean_color = np.mean(img1, axis=(0, 1))
    assert np.all(np.abs(mean_color - [220, 245, 245]) < 2.0)

def test_check_image_folder_completeness_success(temp_workspace):
    workspace, dense_dir, output_dir = temp_workspace
    texturer = OpenMVSTexturer(bin_dir=str(workspace))
    
    # Folder is complete
    texturer._check_image_folder_completeness(dense_dir, dense_dir / "images", MagicMock())

def test_check_image_folder_completeness_failure(temp_workspace):
    workspace, dense_dir, output_dir = temp_workspace
    texturer = OpenMVSTexturer(bin_dir=str(workspace))
    
    # Create incomplete folder
    incomplete_dir = output_dir / "incomplete"
    incomplete_dir.mkdir()
    shutil.copy2(dense_dir / "images" / "frame_0000.jpg", incomplete_dir / "frame_0000.jpg")
    
    with pytest.raises(RuntimeError) as excinfo:
        texturer._check_image_folder_completeness(dense_dir, incomplete_dir, MagicMock())
    
    assert "TEXTURE_IMAGE_FOLDER_INCOMPLETE" in str(excinfo.value)

@patch("modules.reconstruction_engine.openmvs_texturer.OpenMVSTexturer._run_command")
def test_run_texturing_uses_compatible_folder(mock_run, temp_workspace):
    workspace, dense_dir, output_dir = temp_workspace
    texturer = OpenMVSTexturer(bin_dir=str(workspace))
    
    # Mock is_available to return True
    texturer.is_available = MagicMock(return_value=True)
    
    # Mock TextureFrameFilter
    with patch("modules.reconstruction_engine.texture_frame_filter.TextureFrameFilter.filter_session_images") as mock_filter:
        mock_filter.return_value = {
            "selected_images_dir": str(output_dir / "selected"),
            "selected_count": 1,
            "selected_frames": [{"name": "frame_0000.jpg", "path": str(dense_dir / "images" / "frame_0000.jpg")}],
            "fallback_used": False,
            "masked_images_dir": None
        }
        (output_dir / "selected").mkdir()
        
        # We need to mock more things to let it run until InterfaceCOLMAP
        with patch("trimesh.load"):
            try:
                texturer.run_texturing(
                    colmap_workspace=workspace,
                    dense_workspace=dense_dir,
                    selected_mesh="dummy.obj",
                    output_dir=output_dir
                )
            except Exception:
                pass # It will fail later but we check the call
                
    # Check that InterfaceCOLMAP was called with compatible_images
    found_interface_call = False
    for call in mock_run.call_args_list:
        cmd = call[0][0]
        if "InterfaceCOLMAP" in cmd[0]:
            found_interface_call = True
            # Find --image-folder
            for idx, arg in enumerate(cmd):
                if arg == "--image-folder":
                    img_folder = Path(cmd[idx+1])
                    assert img_folder.name == "compatible_images"
                    # Verify it's actually compatible
                    assert (img_folder / "frame_0000.jpg").exists()
                    assert (img_folder / "frame_0001.jpg").exists()
                    assert (img_folder / "frame_0002.jpg").exists()
            break
    assert found_interface_call
