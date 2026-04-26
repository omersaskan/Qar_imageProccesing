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
def workspace_46(tmp_path):
    """Creates a workspace with 46 images."""
    workspace = tmp_path / "workspace_46"
    workspace.mkdir()
    
    images_dir = workspace / "images"
    images_dir.mkdir()
    
    # Create 46 dummy images (black)
    for i in range(46):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"frame_{i:04d}.jpg"), img)
        
    dense_dir = workspace / "dense"
    dense_dir.mkdir()
    (dense_dir / "images").mkdir()
    for i in range(46):
        shutil.copy2(images_dir / f"frame_{i:04d}.jpg", dense_dir / "images" / f"frame_{i:04d}.jpg")
        
    output_dir = workspace / "output"
    output_dir.mkdir()
    
    return workspace, dense_dir, output_dir

def test_compatible_images_full_coverage(workspace_46):
    workspace, dense_dir, output_dir = workspace_46
    texturer = OpenMVSTexturer(bin_dir=str(workspace))
    
    original_images_dir = dense_dir / "images"
    target_dir = output_dir / "compatible_images"
    
    # Only 20 are "selected"
    selected_names = [f"frame_{i:04d}.jpg" for i in range(20)]
    
    texturer._create_compatible_image_folder(
        original_images_dir=original_images_dir,
        target_dir=target_dir,
        selected_names=selected_names,
        log_file=MagicMock()
    )
    
    # 1. Assert compatible_images contains all 46 filenames
    files = list(target_dir.glob("*.jpg"))
    assert len(files) == 46
    
    # 2. Assert selected frames (0-19) are copied from original (black)
    for i in range(20):
        img = cv2.imread(str(target_dir / f"frame_{i:04d}.jpg"))
        assert np.all(img == 0)
        
    # 3. Assert rejected frames (20-45) are neutralized cream
    # Cream color: (220, 245, 245) in BGR
    for i in range(20, 46):
        img = cv2.imread(str(target_dir / f"frame_{i:04d}.jpg"))
        mean_color = np.mean(img, axis=(0, 1))
        assert np.all(np.abs(mean_color - [220, 245, 245]) < 2.0)

@patch("modules.reconstruction_engine.openmvs_texturer.OpenMVSTexturer._run_command")
def test_interface_colmap_uses_compatible_folder(mock_run, workspace_46):
    workspace, dense_dir, output_dir = workspace_46
    texturer = OpenMVSTexturer(bin_dir=str(workspace))
    texturer.is_available = MagicMock(return_value=True)
    
    # Mock filter to return 20 selected frames
    with patch("modules.reconstruction_engine.texture_frame_filter.TextureFrameFilter.filter_session_images") as mock_filter:
        mock_filter.return_value = {
            "selected_images_dir": str(output_dir / "selected"),
            "selected_count": 20,
            "selected_frames": [{"name": f"frame_{i:04d}.jpg", "path": str(dense_dir / "images" / f"frame_{i:04d}.jpg")} for i in range(20)],
            "fallback_used": False,
            "masked_images_dir": None
        }
        (output_dir / "selected").mkdir()
        
        with patch("trimesh.load"):
            try:
                texturer.run_texturing(
                    colmap_workspace=workspace,
                    dense_workspace=dense_dir,
                    selected_mesh="dummy.obj",
                    output_dir=output_dir
                )
            except Exception:
                pass
                
    # 4. Assert InterfaceCOLMAP command uses compatible_images
    found_interface_call = False
    for call in mock_run.call_args_list:
        cmd = call[0][0]
        if "InterfaceCOLMAP" in cmd[0]:
            found_interface_call = True
            for idx, arg in enumerate(cmd):
                if arg == "--image-folder":
                    img_folder = Path(cmd[idx+1])
                    # InterfaceCOLMAP must use compatible_images, not the 20-file selected_images
                    assert "compatible_images" in img_folder.name
                    assert len(list(img_folder.glob("*.jpg"))) == 46
            break
    assert found_interface_call

def test_preflight_incomplete_folder_raises(workspace_46):
    workspace, dense_dir, output_dir = workspace_46
    texturer = OpenMVSTexturer(bin_dir=str(workspace))
    
    # Create folder with only 10 images
    incomplete_dir = output_dir / "incomplete"
    incomplete_dir.mkdir()
    for i in range(10):
        shutil.copy2(dense_dir / "images" / f"frame_{i:04d}.jpg", incomplete_dir / f"frame_{i:04d}.jpg")
    
    # 5. Assert preflight raises TEXTURE_IMAGE_FOLDER_INCOMPLETE
    with pytest.raises(RuntimeError) as excinfo:
        texturer._check_image_folder_completeness(dense_dir, incomplete_dir, MagicMock())
    
    assert "TEXTURE_IMAGE_FOLDER_INCOMPLETE" in str(excinfo.value)
    assert "36 images missing" in str(excinfo.value)

@patch("modules.reconstruction_engine.openmvs_texturer.OpenMVSTexturer._run_command")
def test_top12_retry_still_contains_all_referenced(mock_run, workspace_46):
    workspace, dense_dir, output_dir = workspace_46
    texturer = OpenMVSTexturer(bin_dir=str(workspace))
    texturer.is_available = MagicMock(return_value=True)
    
    with patch("modules.reconstruction_engine.texture_frame_filter.TextureFrameFilter.filter_session_images") as mock_filter:
        mock_filter.return_value = {
            "selected_images_dir": str(output_dir / "selected"),
            "selected_count": 20,
            "selected_frames": [{"name": f"frame_{i:04d}.jpg", "path": str(dense_dir / "images" / f"frame_{i:04d}.jpg")} for i in range(20)],
            "fallback_used": False,
            "masked_images_dir": None
        }
        (output_dir / "selected").mkdir()
        
        with patch("trimesh.load"):
            try:
                # Run with top_n=12
                texturer.run_texturing(
                    colmap_workspace=workspace,
                    dense_workspace=dense_dir,
                    selected_mesh="dummy.obj",
                    output_dir=output_dir,
                    top_n=12
                )
            except Exception:
                pass
                
    # 6. Assert compatible_images_top12 contains ALL 46 filenames, not only 12
    found_interface_call = False
    for call in mock_run.call_args_list:
        cmd = call[0][0]
        if "InterfaceCOLMAP" in cmd[0]:
            found_interface_call = True
            for idx, arg in enumerate(cmd):
                if arg == "--image-folder":
                    img_folder = Path(cmd[idx+1])
                    assert "compatible_images_top12" in img_folder.name
                    # Key requirement: must have 46 files even though only 12 are "active"
                    assert len(list(img_folder.glob("*.jpg"))) == 46
            break
    assert found_interface_call

def test_masked_retry_compatibility(workspace_46):
    """Verifies that image_folder_override (used for masked retries) also triggers compatible folder creation."""
    workspace, dense_dir, output_dir = workspace_46
    texturer = OpenMVSTexturer(bin_dir=str(workspace))
    texturer.is_available = MagicMock(return_value=True)
    
    # Scenario: TexturingService passes a folder with 20 masked images
    masked_dir = output_dir / "selected_images_masked"
    masked_dir.mkdir()
    # Create 20 masked images (blue for testing)
    for i in range(20):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:,:] = [255, 0, 0] # Blue
        cv2.imwrite(str(masked_dir / f"frame_{i:04d}.jpg"), img)
        
    with patch("modules.reconstruction_engine.openmvs_texturer.OpenMVSTexturer._run_command") as mock_run:
        with patch("trimesh.load"):
            try:
                texturer.run_texturing(
                    colmap_workspace=workspace,
                    dense_workspace=dense_dir,
                    selected_mesh="dummy.obj",
                    output_dir=output_dir,
                    image_folder_override=masked_dir
                )
            except Exception:
                pass
                
    # Check InterfaceCOLMAP call
    found_interface_call = False
    for call in mock_run.call_args_list:
        cmd = call[0][0]
        if "InterfaceCOLMAP" in cmd[0]:
            found_interface_call = True
            for idx, arg in enumerate(cmd):
                if arg == "--image-folder":
                    img_folder = Path(cmd[idx+1])
                    # Should be a compatible folder with 46 images
                    assert len(list(img_folder.glob("*.jpg"))) == 46
                    # The 20 masked ones should be blue
                    img0 = cv2.imread(str(img_folder / "frame_0000.jpg"))
                    mean_color = np.mean(img0, axis=(0, 1))
                    assert np.all(np.abs(mean_color - [255, 0, 0]) < 2.0)
                    # The rest should be cream
                    img20 = cv2.imread(str(img_folder / "frame_0020.jpg"))
                    mean_color = np.mean(img20, axis=(0, 1))
                    assert np.all(np.abs(mean_color - [220, 245, 245]) < 2.0)
            break
    assert found_interface_call
