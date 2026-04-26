import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import os

from modules.reconstruction_engine.openmvs_texturer import OpenMVSTexturer
from modules.reconstruction_engine.failures import TexturingFailed
from modules.qa_validation.texture_quality import TextureQualityAnalyzer

@patch("modules.reconstruction_engine.openmvs_texturer.OpenMVSTexturer._run_command")
@patch("modules.reconstruction_engine.openmvs_texturer.OpenMVSTexturer._simplify_mesh")
@patch("modules.reconstruction_engine.texture_frame_filter.TextureFrameFilter.filter_session_images")
def test_texturing_retry_ladder_handles_crash(mock_filter, mock_simplify, mock_run, tmp_path):
    # Setup
    texturer = OpenMVSTexturer(bin_dir=str(tmp_path))
    # Mock binaries
    texturer._interface_colmap.touch()
    texturer._texture_mesh.touch()
    
    out_dir = tmp_path / "texturing_out"
    out_dir.mkdir()
    
    dense_dir = tmp_path / "dense"
    dense_dir.mkdir()
    (dense_dir / "images").mkdir()
    (dense_dir / "images" / "img1.jpg").touch()
    
    mesh_path = tmp_path / "mesh.ply"
    mesh_path.touch()
    
    mock_filter.return_value = {
        "selected_frames": [{"name": "img1.jpg"}],
        "has_masks_available": True,
        "masked_images_dir": str(tmp_path / "masked")
    }

    # Mock _run_command behavior:
    # 1. InterfaceCOLMAP succeeds
    # 2. Attempt A fails with crash code 3221226505
    # 3. Attempt B fails
    # 4. Attempt C succeeds
    
    def side_effect(cmd, cwd, log_file):
        if "InterfaceCOLMAP" in cmd[0]:
            (out_dir / "scene.mvs").touch()
            return
        cmd_str = " ".join(cmd)
        if "textured_model_attempt_a" in cmd_str:
            raise RuntimeError("Command failed with exit code 3221226505: TextureMesh")
        if "textured_model_attempt_b" in cmd_str:
            raise RuntimeError("Command failed with exit code 1: TextureMesh")
        if "textured_model_attempt_c" in cmd_str:
            # Create outputs for Attempt C
            (out_dir / "textured_model_attempt_c.obj").touch()
            (out_dir / "textured_model_attempt_c_map_Kd.png").touch()
            (out_dir / "textured_model_attempt_c.mtl").touch()
            return
    
    mock_run.side_effect = side_effect
    
    result = texturer.run_texturing(
        colmap_workspace=tmp_path / "colmap",
        dense_workspace=dense_dir,
        selected_mesh=str(mesh_path),
        output_dir=out_dir
    )
    
    assert result["attempt_used"] == "textured_model_attempt_c"
    assert "textured_model_attempt_c.obj" in result["textured_mesh_path"]
    # Verify simplification was called (Attempt A, B, C use simplified meshes)
    assert mock_simplify.call_count >= 1

@patch("modules.reconstruction_engine.openmvs_texturer.OpenMVSTexturer._run_command")
@patch("modules.reconstruction_engine.texture_frame_filter.TextureFrameFilter.filter_session_images")
def test_has_masks_available_false_generates_guidance(mock_filter, mock_run, tmp_path):
    texturer = OpenMVSTexturer(bin_dir=str(tmp_path))
    texturer._interface_colmap.touch()
    texturer._texture_mesh.touch()
    
    out_dir = tmp_path / "texturing_out"
    out_dir.mkdir()
    
    dense_dir = tmp_path / "dense"
    dense_dir.mkdir()
    (dense_dir / "images").mkdir()
    (dense_dir / "images" / "img1.jpg").touch()
    
    mesh_path = tmp_path / "mesh.ply"
    mesh_path.touch()
    
    mock_filter.return_value = {
        "selected_frames": [{"name": "img1.jpg"}],
        "has_masks_available": False,
        "masked_images_dir": None
    }
    
    def side_effect(cmd, cwd, log_file):
        if "InterfaceCOLMAP" in cmd[0]:
            (out_dir / "scene.mvs").touch()
            return
        if "textured_model_attempt_a" in " ".join(cmd):
            (out_dir / "textured_model_attempt_a.obj").touch()
            (out_dir / "textured_model_attempt_a_map_Kd.png").touch()
            return
            
    mock_run.side_effect = side_effect
    
    result = texturer.run_texturing(
        colmap_workspace=tmp_path / "colmap",
        dense_workspace=dense_dir,
        selected_mesh=str(mesh_path),
        output_dir=out_dir
    )
    
    assert result["has_masks_available"] is False
    assert "Mask unavailable" in result["operator_guidance"]

def test_texture_quality_threshold_reporting():
    # Mock settings
    mock_settings = MagicMock()
    mock_settings.white_cream_max_background_ratio = 0.18
    mock_settings.max_dominant_background_ratio = 0.5
    mock_settings.max_flat_color_ratio = 0.7
    mock_settings.max_black_pixel_ratio = 0.4
    mock_settings.max_near_black_ratio = 0.6
    mock_settings.min_atlas_coverage_ratio = 0.3
    
    analyzer = TextureQualityAnalyzer(thresholds=mock_settings)
    
    # Create a dummy 3-channel image
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    
    # Case 1: white_cream
    res1 = analyzer.analyze_image(dummy_img, expected_product_color="white_cream")
    assert res1["background_threshold_used"] == 0.18
    
    # Case 2: default
    res2 = analyzer.analyze_image(dummy_img, expected_product_color="unknown")
    assert res2["background_threshold_used"] == 0.5

def test_exit_code_3221226505_mapping():
    from modules.reconstruction_engine.failures import TexturingFailed
    err = TexturingFailed("Crash", exit_code=3221226505)
    assert "TEXTUREMESH_NATIVE_CRASH" in str(err)
    assert "3221226505" in str(err)
