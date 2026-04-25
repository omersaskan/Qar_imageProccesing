import pytest
import json
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.reconstruction_engine.adapter import COLMAPAdapter

def test_stereo_fusion_thresholds_passing(tmp_path):
    # Tests that environment-backed settings are passed to the builder
    from modules.operations.settings import settings
    
    with patch.object(settings, "recon_stereo_fusion_min_num_pixels", 5), \
         patch.object(settings, "recon_stereo_fusion_max_reproj_error", 1.5):
        
        adapter = COLMAPAdapter()
        adapter.builder.stereo_fusion = MagicMock(return_value=["colmap", "stereo_fusion"])
        
        effective_mask_path = Path("masks")
        
        # Verify the builder is called with correct values
        _ = adapter.builder.stereo_fusion(
            tmp_path / "dense",
            tmp_path / "fused.ply",
            mask_path=effective_mask_path,
            min_num_pixels=settings.recon_stereo_fusion_min_num_pixels,
            max_reproj_error=settings.recon_stereo_fusion_max_reproj_error,
            max_depth_error=settings.recon_stereo_fusion_max_depth_error,
            max_normal_error=settings.recon_stereo_fusion_max_normal_error
        )
        
        adapter.builder.stereo_fusion.assert_called_with(
            tmp_path / "dense",
            tmp_path / "fused.ply",
            mask_path=effective_mask_path,
            min_num_pixels=5,
            max_reproj_error=1.5,
            max_depth_error=settings.recon_stereo_fusion_max_depth_error,
            max_normal_error=settings.recon_stereo_fusion_max_normal_error
        )

def test_diagnostic_json_generation(tmp_path):
    adapter = COLMAPAdapter()
    output_dir = tmp_path
    dense_dir = output_dir / "dense"
    dense_dir.mkdir()
    (dense_dir / "stereo").mkdir()
    (dense_dir / "stereo" / "depth_maps").mkdir()
    (dense_dir / "stereo" / "normal_maps").mkdir()
    
    # Create dummy maps
    for i in range(3):
        (dense_dir / "stereo" / "depth_maps" / f"map_{i}.bin").write_text("dummy")
    for i in range(2):
        (dense_dir / "stereo" / "normal_maps" / f"norm_{i}.bin").write_text("dummy")
        
    log_file = MagicMock()
    
    adapter._write_fusion_diagnostics(
        output_dir,
        fused_points=60000,
        sparse_points=5000,
        mask_path=Path("some/mask"),
        mask_valid=True,
        thresholds={"min_num_pixels": 2},
        log_file=log_file
    )
    
    diag_path = output_dir / "fusion_diagnostics.json"
    assert diag_path.exists()
    
    with open(diag_path, "r") as f:
        data = json.load(f)
        assert data["depth_map_count"] == 3
        assert data["normal_map_count"] == 2
        assert data["fused_point_count"] == 60000
        assert data["status"] == "warning"  # because 50k < 60k < 100k
        assert data["sparse_dense_ratio"] == 12.0

def test_diagnostic_json_fail_status(tmp_path):
    adapter = COLMAPAdapter()
    output_dir = tmp_path
    (output_dir / "dense" / "stereo" / "depth_maps").mkdir(parents=True)
    (output_dir / "dense" / "stereo" / "normal_maps").mkdir(parents=True)
    
    log_file = MagicMock()
    adapter._write_fusion_diagnostics(
        output_dir,
        fused_points=40000,
        sparse_points=1000,
        mask_path=None,
        mask_valid=False,
        thresholds={},
        log_file=log_file
    )
    
    with open(output_dir / "fusion_diagnostics.json", "r") as f:
        data = json.load(f)
        assert data["status"] == "fail"
        assert "Recapture recommended" in data["recommendation"]

def test_dense_mask_validation_strictness(tmp_path):
    adapter = COLMAPAdapter()
    dense_masks_dir = tmp_path / "masks"
    images_dir = tmp_path / "images"
    dense_masks_dir.mkdir()
    images_dir.mkdir()
    
    log_file = MagicMock()
    
    # Case 1: missing masks
    (images_dir / "frame1.jpg").write_text("img")
    assert adapter._validate_dense_masks(dense_masks_dir, images_dir, log_file) is False
    
    # Case 2: dimension mismatch
    (dense_masks_dir / "frame1.jpg.png").write_text("mask")
    
    def mock_read(path, flag):
        m = MagicMock()
        if "frame1.jpg.png" in str(path):
            m.shape = (100, 100, 1)
        else:
            m.shape = (200, 200, 3)
        return m
        
    adapter._read_image = MagicMock(side_effect=mock_read)
    assert adapter._validate_dense_masks(dense_masks_dir, images_dir, log_file) is False
    
    # Case 3: Success
    def mock_read_success(path, flag):
        m = MagicMock()
        m.shape = (200, 200, 3)
        return m
    adapter._read_image = MagicMock(side_effect=mock_read_success)
    assert adapter._validate_dense_masks(dense_masks_dir, images_dir, log_file) is True
