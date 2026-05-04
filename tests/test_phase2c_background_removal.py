import pytest
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch
from modules.ai_3d_generation.input_preprocessor import preprocess_input

def test_bg_removal_disabled_by_default(tmp_path):
    src_img = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(src_img)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    res = preprocess_input(str(src_img), str(out_dir), background_removal_enabled=False)
    assert res["background_removed"] is False
    assert res["mask_source"] in ("none", "fallback_center_crop")

def test_bg_removal_rembg_unavailable(tmp_path):
    src_img = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(src_img)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    # Force import error
    with patch.dict(sys.modules, {'rembg': None}):
        res = preprocess_input(str(src_img), str(out_dir), background_removal_enabled=True)
    
    assert res["background_removed"] is False
    assert any("rembg_unavailable" in w for w in res["warnings"])
    assert res["mask_source"] == "fallback_center_crop"

def test_bg_removal_success(tmp_path):
    src_img = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100), color="blue").save(src_img)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    # Mock rembg
    mock_rembg = MagicMock()
    # Return 100x100 RGBA with 40x40 alpha square in the middle
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[30:70, 30:70, 3] = 255
    mock_rembg.remove.return_value = rgba
    
    with patch.dict(sys.modules, {'rembg': mock_rembg}):
        res = preprocess_input(str(src_img), str(out_dir), background_removal_enabled=True)
    
    assert res["background_removed"] is True
    assert res["mask_source"] == "rembg"
    assert res["bbox_source"] == "rembg_alpha"
    assert res["alpha_bbox"] == [30, 30, 69, 69]
    assert res["foreground_ratio_estimate"] > 0
    assert res["bbox"] == res["alpha_bbox"]
    
    # Verify file is RGBA
    out_img = Image.open(res["prepared_image_path"])
    assert out_img.mode == "RGBA"

def test_bg_removal_empty_alpha_fallback(tmp_path):
    src_img = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(src_img)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    mock_rembg = MagicMock()
    # Return empty alpha
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    mock_rembg.remove.return_value = rgba
    
    with patch.dict(sys.modules, {'rembg': mock_rembg}):
        res = preprocess_input(str(src_img), str(out_dir), background_removal_enabled=True)
        
    assert res["background_removed"] is False
    assert any("rembg_empty_alpha" in w for w in res["warnings"])
    assert res["mask_source"] == "fallback_center_crop"

def test_pipeline_bg_removal_propagation(tmp_path):
    from modules.ai_3d_generation.pipeline import generate_ai_3d
    
    sess_id = "test_p2c_pipeline"
    output_dir = tmp_path / sess_id
    output_dir.mkdir()
    
    input_img = tmp_path / "input.jpg"
    Image.new("RGB", (100, 100)).save(input_img)
    
    with patch("modules.ai_3d_generation.pipeline._get_provider") as mock_get_p, \
         patch("modules.ai_3d_generation.pipeline.run_postprocess"), \
         patch("modules.ai_3d_generation.pipeline.quality_evaluate") as mock_eval, \
         patch("modules.ai_3d_generation.pipeline.preprocess_input") as mock_prep, \
         patch("modules.operations.settings.settings.ai_3d_multi_candidate_enabled", False):
        
        mock_p = MagicMock()
        mock_p.name = "sf3d"
        mock_p.output_format = "glb"
        mock_p.safe_generate.return_value = {"status": "ok", "output_path": "out.glb", "metadata": {}}
        mock_get_p.return_value = mock_p
        mock_eval.return_value = {"verdict": "ok", "warnings": []}
        mock_prep.return_value = {"enabled": True, "prepared_image_path": "fake.png", "warnings": []}
        
        # 1. Test via options
        generate_ai_3d(sess_id, str(input_img), str(output_dir), "sf3d", {"background_removal_enabled": True})
        # Check that mock_prep was called with background_removal_enabled=True
        mock_prep.assert_called_with(
            source_image_path=mock_prep.call_args.kwargs["source_image_path"],
            output_dir=mock_prep.call_args.kwargs["output_dir"],
            input_size=mock_prep.call_args.kwargs["input_size"],
            bbox_padding_ratio=mock_prep.call_args.kwargs["bbox_padding_ratio"],
            background_removal_enabled=True
        )

def test_candidate_runner_bg_removal_propagation(tmp_path):
    from modules.ai_3d_generation.candidate_runner import run_candidates_sequential
    
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "derived" / "candidates").mkdir(parents=True)
    
    src_img = tmp_path / "input.jpg"
    Image.new("RGB", (100, 100)).save(src_img)
    
    mock_provider = MagicMock()
    mock_provider.safe_generate.return_value = {"status": "ok", "output_path": "out.glb", "metadata": {}}
    
    with patch("modules.ai_3d_generation.candidate_runner.preprocess_input") as mock_prep:
        mock_prep.return_value = {"enabled": True, "prepared_image_path": "fake.png", "warnings": []}
        
        run_candidates_sequential(
            session_dir=str(session_dir),
            source_paths=[str(src_img)],
            provider=mock_provider,
            max_candidates=1,
            background_removal_enabled=True
        )
        
        assert mock_prep.call_args.kwargs["background_removal_enabled"] is True
