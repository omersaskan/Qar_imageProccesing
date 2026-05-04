import pytest
import os
from pathlib import Path
from PIL import Image
from modules.ai_3d_generation.input_preprocessor import preprocess_input

def test_preprocess_rich_metadata(tmp_path):
    # Create a test image
    src_img = tmp_path / "test.jpg"
    Image.new("RGB", (100, 200), color="red").save(src_img)
    
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    input_size = 512
    res = preprocess_input(str(src_img), str(out_dir), input_size=input_size, bbox_padding_ratio=0.15)
    
    # Check required fields
    required_fields = [
        "enabled", "source_image_path", "prepared_image_path", "input_size",
        "original_width", "original_height", "output_width", "output_height",
        "crop_method", "bbox", "bbox_padding_ratio", "background_removed",
        "mask_source", "alpha_bbox", "foreground_ratio_estimate", "warnings"
    ]
    for field in required_fields:
        assert field in res, f"Missing field: {field}"
        
    assert res["enabled"] is True
    assert res["input_size"] == input_size
    assert res["original_width"] == 100
    assert res["original_height"] == 200
    assert res["output_width"] == input_size
    assert res["output_height"] == input_size
    assert res["background_removed"] is False
    assert res["mask_source"] == "fallback_center_crop"
    assert isinstance(res["bbox"], list)
    assert len(res["bbox"]) == 4
    assert res["bbox_padding_ratio"] == 0.15
    assert Path(res["prepared_image_path"]).exists()

def test_preprocess_crop_methods(tmp_path):
    src_img = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(src_img)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    # 1. Fallback
    res = preprocess_input(str(src_img), str(out_dir))
    assert res["crop_method"] == "fallback_center_crop"
    
    # 2. BBox
    res = preprocess_input(str(src_img), str(out_dir), bbox=(10, 10, 50, 50))
    assert res["crop_method"] == "center_square_crop"
    
    # 3. Mask (empty)
    import numpy as np
    mask = np.zeros((100, 100), dtype=np.uint8)
    res = preprocess_input(str(src_img), str(out_dir), mask=mask)
    assert res["crop_method"] == "resize_square_pad"
    assert any("mask_empty" in w for w in res["warnings"])

def test_pipeline_manifest_includes_rich_preprocessing(tmp_path):
    from modules.ai_3d_generation.pipeline import generate_ai_3d
    from unittest.mock import patch, MagicMock
    
    sess_id = "test_p2b_manifest"
    output_dir = tmp_path / sess_id
    output_dir.mkdir()
    
    input_img = tmp_path / "input.jpg"
    Image.new("RGB", (100, 100)).save(input_img)
    
    with patch("modules.ai_3d_generation.pipeline._get_provider") as mock_get_p, \
         patch("modules.ai_3d_generation.pipeline.run_postprocess") as mock_post, \
         patch("modules.ai_3d_generation.pipeline.quality_evaluate") as mock_eval, \
         patch("modules.operations.settings.settings.ai_3d_multi_candidate_enabled", False):
        
        mock_p = MagicMock()
        mock_p.name = "sf3d"
        mock_p.license_note = "test"
        mock_p.output_format = "glb"
        mock_p.safe_generate.return_value = {
            "status": "ok",
            "output_path": str(output_dir / "out.glb"),
            "metadata": {}
        }
        mock_get_p.return_value = mock_p
        mock_eval.return_value = {"verdict": "ok", "warnings": []}
        
        # Create dummy GLB
        (output_dir / "out.glb").write_bytes(b"dummy")
        
        manifest = generate_ai_3d(sess_id, str(input_img), str(output_dir), "sf3d", {"quality_mode": "ultra"})
        
        assert "preprocessing" in manifest
        pre = manifest["preprocessing"]
        assert pre["enabled"] is True
        assert "original_width" in pre
        assert "bbox" in pre
        assert pre["bbox_padding_ratio"] == 0.14  # Ultra default from previous phase
        assert pre["background_removed"] is False
