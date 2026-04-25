import pytest
import numpy as np
import cv2
import json
import io
from pathlib import Path
from modules.qa_validation.validator import AssetValidator
from modules.qa_validation.rules import ValidationThresholds
from modules.operations.guidance import GuidanceAggregator
from modules.reconstruction_engine.adapter import OpenMVSAdapter
from modules.operations.settings import settings

def test_validator_with_texture_path_does_not_crash(tmp_path):
    # Create a dummy texture
    tex_path = tmp_path / "test_texture.png"
    img = np.full((100, 100, 3), 200, dtype=np.uint8) # Clean gray
    cv2.imwrite(str(tex_path), img)
    
    validator = AssetValidator(thresholds=ValidationThresholds())
    val_input = {
        "poly_count": 1000,
        "texture_path": str(tex_path),
        "expected_product_color": "unknown",
        "bbox": {"x": 10.0, "y": 10.0, "z": 10.0},
        "ground_offset": 0.0,
        "cleanup_stats": {},
        "texture_path_exists": True,
        "has_uv": True,
        "has_material": True,
        "has_embedded_texture": False,
        "delivery_geometry_count": 1,
        "delivery_component_count": 1
    }
    
    report = validator.validate("test_asset", val_input)
    assert report.texture_quality_status == "clean"

def test_contaminated_texture_fails_validation(tmp_path):
    # Create a contaminated texture (heavy black)
    tex_path = tmp_path / "bad_texture.png"
    img = np.zeros((100, 100, 3), dtype=np.uint8) # All black
    cv2.imwrite(str(tex_path), img)
    
    validator = AssetValidator(thresholds=ValidationThresholds())
    val_input = {
        "poly_count": 1000,
        "texture_path": str(tex_path),
        "expected_product_color": "unknown",
        "bbox": {"x": 10.0, "y": 10.0, "z": 10.0},
        "ground_offset": 0.0,
        "cleanup_stats": {},
        "texture_path_exists": True,
        "has_uv": True,
        "has_material": True,
        "has_embedded_texture": False,
        "delivery_geometry_count": 1,
        "delivery_component_count": 1
    }
    
    report = validator.validate("test_asset", val_input)
    assert report.texture_quality_status == "contaminated"
    assert report.final_decision == "fail"

def test_clean_white_texture_with_profile_passes(tmp_path):
    # Create a clean white texture
    tex_path = tmp_path / "white_texture.png"
    img = np.full((100, 100, 3), 240, dtype=np.uint8)
    cv2.imwrite(str(tex_path), img)
    
    validator = AssetValidator(thresholds=ValidationThresholds())
    val_input = {
        "poly_count": 1000,
        "texture_path": str(tex_path),
        "expected_product_color": "white_cream",
        "bbox": {"x": 10.0, "y": 10.0, "z": 10.0},
        "ground_offset": 0.0,
        "cleanup_stats": {},
        "texture_path_exists": True,
        "has_uv": True,
        "has_material": True,
        "has_embedded_texture": False,
        "delivery_geometry_count": 1,
        "delivery_component_count": 1
    }
    
    report = validator.validate("test_asset", val_input)
    assert report.texture_quality_status == "clean"
    assert report.near_white_ratio > 0.9

def test_guidance_reads_texture_quality_reasons():
    aggregator = GuidanceAggregator()
    val_report = {
        "final_decision": "fail",
        "texture_quality_status": "contaminated",
        "texture_quality_reasons": ["BLACK_PATCH_RATIO_HIGH", "BACKGROUND_COLOR_CONTAMINATION"]
    }
    
    guidance = aggregator.generate_guidance(
        session_id="test_session",
        status="validated",
        validation_report=val_report
    )
    
    codes = [m["code"] for m in guidance.messages]
    assert "TEXTURE_ATLAS_CONTAMINATION" in codes
    # Check if specific reason matched
    assert "TEXTURE_ATLAS_CONTAMINATION" in codes # match_failure_reason maps them to this code

def test_mask_refinement_key_compatibility(tmp_path):
    # Setup mock structure
    ws = tmp_path / "ws"
    ws.mkdir()
    masks_dir = ws / "masks"
    masks_dir.mkdir()
    
    capture_dir = tmp_path / "capture"
    frames_dir = capture_dir / "frames"
    frames_dir.mkdir(parents=True)
    orig_meta_dir = frames_dir / "masks"
    orig_meta_dir.mkdir()
    
    # 1. Test subject_clipped
    frame_clipped = frames_dir / "clipped.jpg"
    cv2.imwrite(str(frame_clipped), np.zeros((10,10,3)))
    mask_clipped = masks_dir / "clipped.jpg.png"
    cv2.imwrite(str(mask_clipped), np.full((10,10), 255, dtype=np.uint8))
    
    with open(orig_meta_dir / "clipped.json", "w") as f:
        json.dump({"subject_clipped": True}, f)
        
    # 2. Test support_contamination_detected in reasons
    frame_support = frames_dir / "support.jpg"
    cv2.imwrite(str(frame_support), np.zeros((10,10,3)))
    mask_support = masks_dir / "support.jpg.png"
    cv2.imwrite(str(mask_support), np.full((10,10), 255, dtype=np.uint8))
    
    with open(orig_meta_dir / "support.json", "w") as f:
        json.dump({"failure_reasons": ["support_contamination_detected"]}, f)
        
    adapter = OpenMVSAdapter()
    log = io.StringIO()
    prep = {"masks_dir": masks_dir}
    input_frames = [str(frame_clipped), str(frame_support)]
    
    settings.texture_reject_subject_clipped = True
    settings.texture_reject_support_contamination = True
    settings.texture_min_clean_frames = 0
    
    adapter._refine_texture_masks(prep, input_frames, log)
    
    # Both should be blanked
    m1 = cv2.imread(str(mask_clipped), cv2.IMREAD_GRAYSCALE)
    assert np.sum(m1 > 0) == 0
    
    m2 = cv2.imread(str(mask_support), cv2.IMREAD_GRAYSCALE)
    assert np.sum(m2 > 0) == 0
