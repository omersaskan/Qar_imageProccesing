import pytest
import cv2
import numpy as np
from pathlib import Path
from modules.reconstruction_engine.texture_frame_filter import TextureFrameFilter
from modules.operations.atlas_repair_service import AtlasRepairService
from modules.qa_validation.texture_quality import TextureQualityAnalyzer

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

def test_texture_frame_filter_rejects_dark_and_blurry(temp_dir):
    image_dir = temp_dir / "images"
    image_dir.mkdir()
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    
    # 1. Dark image
    dark_path = image_dir / "dark.jpg"
    cv2.imwrite(str(dark_path), np.zeros((100, 100, 3), dtype=np.uint8))
    
    # 2. Blurry image (low variance)
    blurry_path = image_dir / "blurry.jpg"
    cv2.imwrite(str(blurry_path), np.full((100, 100, 3), 128, dtype=np.uint8))
    
    # 3. Good image (High contrast, centered content)
    good_path = image_dir / "good.jpg"
    good_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # White square in the middle
    good_img[20:80, 20:80] = 200
    cv2.imwrite(str(good_path), good_img)
    
    # 4. Mask image (should be rejected by name)
    mask_path = image_dir / "frame_mask.png"
    cv2.imwrite(str(mask_path), np.zeros((100, 100, 3), dtype=np.uint8))
    
    filter = TextureFrameFilter()
    # Ensure it doesn't fallback by making sure good.jpg is actually good
    results = filter.filter_session_images(image_dir, output_dir)
    
    selected_files = [f.name for f in results["selected_images_dir"].iterdir()]
    assert "good.jpg" in selected_files
    assert "dark.jpg" not in selected_files
    assert "blurry.jpg" not in selected_files
    assert "frame_mask.png" not in selected_files
    assert results["selected_count"] == 1

def test_atlas_repair_improves_dark_atlas(temp_dir):
    # 1. Create a dark atlas
    atlas_path = temp_dir / "dark_atlas.jpg"
    # Very dark grey
    img = np.full((512, 512, 3), 30, dtype=np.uint8)
    cv2.imwrite(str(atlas_path), img)
    
    # Use loose thresholds so it passes after repair
    from modules.qa_validation.rules import ValidationThresholds
    loose_thresholds = ValidationThresholds(max_black_pixel_ratio=0.8)
    repair_service = AtlasRepairService(analyzer=TextureQualityAnalyzer(thresholds=loose_thresholds))
    
    results = repair_service.repair_atlas(str(atlas_path), expected_color="white_cream")
    
    assert results["status"] == "repaired"
    assert Path(results["repaired_path"]).exists()
    assert results["repaired_stats"]["average_luminance"] > results["original_stats"]["average_luminance"]

def test_atlas_repair_does_not_fix_neon_contamination(temp_dir):
    # 1. Create a neon contaminated atlas
    atlas_path = temp_dir / "neon_atlas.jpg"
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Add large neon green block (to ensure it triggers neon artifact reason)
    img[0:300, 0:300] = [0, 255, 0]
    cv2.imwrite(str(atlas_path), img)
    
    repair_service = AtlasRepairService()
    results = repair_service.repair_atlas(str(atlas_path))
    
    # Should skip repair because it's contaminated
    assert results["status"] == "repair_skipped"

def test_white_cream_stricter_thresholds():
    from modules.qa_validation.rules import ValidationThresholds
    # Use standard default thresholds (0.40 black)
    std_thresholds = ValidationThresholds(max_black_pixel_ratio=0.40, max_dominant_background_ratio=0.50)
    analyzer = TextureQualityAnalyzer(thresholds=std_thresholds)
    
    # An atlas with 0.30 black ratio (fails white_cream, passes generic)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[0:70, :] = 200 # 70% light, 30% black
    
    # Generic check (30% black < 40% threshold)
    res_generic = analyzer.analyze_image(img, expected_product_color="unknown")
    assert res_generic["texture_quality_status"] == "success"
    
    # white_cream check (threshold 0.20)
    res_white = analyzer.analyze_image(img, expected_product_color="white_cream")
    assert res_white["texture_quality_status"] == "fail"
    assert any("High black pixel ratio" in r for r in res_white["texture_quality_reasons"])
