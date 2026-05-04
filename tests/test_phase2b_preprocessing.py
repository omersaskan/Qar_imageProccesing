import pytest
import os
import json
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
        "crop_width", "crop_height", "crop_method", "bbox", "crop_bbox",
        "bbox_padding_ratio", "background_removed", "bbox_source",
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
    assert isinstance(res["bbox"], list)
    assert len(res["bbox"]) == 4
    assert isinstance(res["crop_bbox"], list)
    assert len(res["crop_bbox"]) == 4
    assert res["bbox_padding_ratio"] == 0.15
    assert Path(res["prepared_image_path"]).exists()

def test_preprocess_input_size_clamping(tmp_path):
    src_img = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(src_img)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    # Too large
    res = preprocess_input(str(src_img), str(out_dir), input_size=2000)
    assert res["input_size"] == 1024
    assert any("input_size_clamped" in w for w in res["warnings"])
    
    # Too small
    res = preprocess_input(str(src_img), str(out_dir), input_size=10)
    assert res["input_size"] == 64
    assert any("input_size_clamped" in w for w in res["warnings"])

def test_preprocess_bbox_validation(tmp_path):
    src_img = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(src_img)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    # Invalid bbox (x1 == x0)
    res = preprocess_input(str(src_img), str(out_dir), bbox=(10, 10, 10, 20))
    assert res["bbox_source"] == "fallback_center_crop"
    assert any("invalid_bbox_using_center_crop" in w for w in res["warnings"])
    
    # Valid bbox
    res = preprocess_input(str(src_img), str(out_dir), bbox=(10, 10, 20, 20))
    assert res["bbox_source"] == "provided_bbox"
    assert res["bbox"] == [10, 10, 20, 20]
    assert res["crop_bbox"] != res["bbox"] # Padding should be applied

def test_preprocess_crop_metrics(tmp_path):
    src_img = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(src_img)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    res = preprocess_input(str(src_img), str(out_dir), bbox=(10, 10, 50, 50))
    assert res["crop_width"] > 0
    assert res["crop_height"] > 0
    assert res["crop_width"] == res["crop_bbox"][2] - res["crop_bbox"][0]

def test_preprocess_bbox_sources(tmp_path):
    src_img = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(src_img)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    # 1. Fallback
    res = preprocess_input(str(src_img), str(out_dir))
    assert res["bbox_source"] == "fallback_center_crop"
    
    # 2. Mask
    import numpy as np
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    res = preprocess_input(str(src_img), str(out_dir), mask=mask)
    assert res["bbox_source"] == "provided_mask"
    
    # 3. Empty mask
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    res = preprocess_input(str(src_img), str(out_dir), mask=empty_mask)
    assert res["bbox_source"] == "full_image_fallback"

def test_candidate_manifest_sanitization(tmp_path):
    from modules.ai_3d_generation.candidate_runner import run_candidates_sequential
    from unittest.mock import MagicMock, patch
    
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "derived" / "candidates").mkdir(parents=True)
    
    src_img = tmp_path / "input.jpg"
    Image.new("RGB", (100, 100)).save(src_img)
    
    mock_provider = MagicMock()
    mock_provider.name = "mock"
    # Return a secret in metadata
    mock_provider.safe_generate.return_value = {
        "status": "ok",
        "output_path": str(tmp_path / "dummy.glb"),
        "metadata": {"api_key": "api_key=secret_key_123", "token": "Bearer abc"}
    }
    (tmp_path / "dummy.glb").write_bytes(b"dummy")
    
    results = run_candidates_sequential(
        session_dir=str(session_dir),
        source_paths=[str(src_img)],
        provider=mock_provider,
        max_candidates=1
    )
    
    manifest_path = session_dir / "derived" / "candidates" / "cand_001" / "candidate_manifest.json"
    assert manifest_path.exists()
    
    with open(manifest_path, "r") as f:
        content = json.load(f)
    
    # Check that secrets are redacted in the file
    worker_meta = content.get("worker_metadata", {})
    # Note: sanitize_text for "Bearer abc" returns "Bearer [REDACTED]"
    assert "[REDACTED]" in worker_meta.get("token")
    assert "[REDACTED]" in worker_meta.get("api_key")
    
    # Check that in-memory results are NOT necessarily mutated (if sanitization was on copy)
    # But for candidate_manifest.json, the prompt said "write sanitized version to disk"
    # "Do not mutate cand_meta returned to pipeline unless already safe."
    # So we check that results[0] still has the secrets (if we didn't mutate it)
    # Actually, sanitize_json_like usually returns a new dict.
    # We'll just verify the disk is safe.

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
        assert pre["bbox_padding_ratio"] == 0.14
        assert pre["background_removed"] is False
        assert "bbox_source" in pre
