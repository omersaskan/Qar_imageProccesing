import cv2
import numpy as np
import pytest
import os
from pathlib import Path
from modules.qa_validation.texture_quality import TextureQualityAnalyzer

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

def test_black_atlas_fails(temp_dir):
    # 1. Create black image
    img_path = temp_dir / "black.jpg"
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    
    # 2. Analyze
    analyzer = TextureQualityAnalyzer()
    results = analyzer.analyze_path(str(img_path), expected_product_color="white_cream")
    
    # 3. Assert
    assert results["texture_quality_status"] == "fail"
    assert results["black_pixel_ratio"] > 0.9
    assert results["texture_quality_grade"] == "F"
    assert any("High black pixel ratio" in r for r in results["texture_quality_reasons"])

def test_neon_atlas_fails(temp_dir):
    # 1. Create neon image (High saturation, High value)
    img_path = temp_dir / "neon.jpg"
    # HSV: Hue=120 (green), Saturation=255, Value=255
    hsv = np.zeros((512, 512, 3), dtype=np.uint8)
    hsv[:, :, 0] = 60 # OpenCV Hue is 0-180
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(img_path), img)
    
    # 2. Analyze
    analyzer = TextureQualityAnalyzer()
    results = analyzer.analyze_path(str(img_path), expected_product_color="white_cream")
    
    # 3. Assert
    assert results["texture_quality_status"] == "fail"
    assert results["neon_artifact_ratio"] > 0.9
    assert results["texture_quality_grade"] == "F"
    assert any("Neon artifacts detected" in r for r in results["texture_quality_reasons"])

def test_white_cream_atlas_passes(temp_dir):
    # 1. Create white/cream image
    img_path = temp_dir / "cream.jpg"
    # Cream: BGR (240, 245, 250)
    img = np.full((512, 512, 3), (240, 245, 250), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    
    # 2. Analyze
    analyzer = TextureQualityAnalyzer()
    results = analyzer.analyze_path(str(img_path), expected_product_color="white_cream")
    
    # 3. Assert
    assert results["texture_quality_status"] == "success"
    assert results["near_white_ratio"] > 0.9
    assert results["texture_quality_grade"] in ["A", "B"]
    assert len(results["texture_quality_reasons"]) == 0

def test_visual_debug_artifacts_generated(temp_dir):
    # 1. Create image
    img_path = temp_dir / "test_debug.jpg"
    img = np.full((512, 512, 3), (128, 128, 128), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    
    # 2. Analyze
    analyzer = TextureQualityAnalyzer()
    analyzer.analyze_path(str(img_path))
    
    # 3. Assert artifacts exist
    assert (temp_dir / "texture_atlas_preview.png").exists()
    assert (temp_dir / "texture_atlas_histogram.json").exists()
