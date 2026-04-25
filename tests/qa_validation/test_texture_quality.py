import pytest
import numpy as np
import cv2
from modules.qa_validation.texture_quality import TextureQualityAnalyzer
from modules.operations.settings import settings

def test_texture_quality_clean_white():
    # Create a 100x100 white image with alpha
    img = np.full((100, 100, 4), 255, dtype=np.uint8)
    # White product: RGB = (240, 240, 240)
    img[:, :, :3] = 240
    
    analyzer = TextureQualityAnalyzer()
    results = analyzer.analyze_image(img, expected_product_color="white_cream")
    
    assert results["texture_quality_status"] == "clean"
    assert results["texture_quality_grade"] == "A"
    assert results["near_white_ratio"] > 0.9

def test_texture_quality_black_heavy_fail():
    # 50% black image
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img[:50, :, :] = 10 # black
    
    analyzer = TextureQualityAnalyzer()
    results = analyzer.analyze_image(img, expected_product_color="unknown")
    
    assert results["texture_quality_status"] == "contaminated"
    assert "BLACK_PATCH_RATIO_HIGH" in results["texture_quality_reasons"]

def test_texture_quality_dark_product_pass():
    # 50% black image but expected "dark"
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img[:50, :, :] = 10 # black
    
    analyzer = TextureQualityAnalyzer()
    results = analyzer.analyze_image(img, expected_product_color="dark")
    
    # Should not fail for black ratio
    assert results["texture_quality_status"] == "clean"

def test_texture_quality_orange_contamination():
    # 30% orange contamination
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    # Orange in BGR: approximately (0, 165, 255)
    # OpenCV Hue for orange is around 10-20. 
    # BGR [0, 165, 255] -> H: 20, S: 255, V: 255 (approx)
    img[:30, :, :] = [0, 165, 255]
    
    analyzer = TextureQualityAnalyzer()
    results = analyzer.analyze_image(img)
    
    assert results["texture_quality_status"] == "contaminated"
    assert "BACKGROUND_COLOR_CONTAMINATION" in results["texture_quality_reasons"]

def test_texture_quality_alpha_handling():
    # 100x100 image, 50% transparent
    img = np.full((100, 100, 4), 255, dtype=np.uint8)
    img[:50, :, 3] = 0 # Transparent
    # Visible part is black
    img[50:, :, :3] = 10
    
    analyzer = TextureQualityAnalyzer()
    results = analyzer.analyze_image(img, expected_product_color="dark")
    
    # Visible pixels are 5000. Black pixels are 5000. Ratio = 1.0.
    # But it's a dark product so it's ok.
    assert results["black_pixel_ratio"] > 0.99
    # Atlas coverage should be 0.5
    assert results["atlas_coverage_ratio"] == 0.5
    assert "LOW_ATLAS_COVERAGE" in results["texture_quality_reasons"]

def test_texture_quality_colorful_product_no_white_fail():
    # Red image (colorful)
    # BGR [0, 0, 255] -> H: 0 (or 180), S: 255, V: 255
    img = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)
    
    analyzer = TextureQualityAnalyzer()
    results = analyzer.analyze_image(img, expected_product_color="colorful")
    
    # Should not fail because of low near_white_ratio
    assert "WHITE_PRODUCT_TEXTURE_RATIO_LOW" not in results["texture_quality_reasons"]
    # It should not fail for background contamination if it's not orange
    assert results["texture_quality_status"] != "contaminated"

def test_texture_quality_white_product_low_white_fail():
    # Expected white product, but texture is gray/colorful
    img = np.full((100, 100, 3), [100, 100, 100], dtype=np.uint8)
    
    analyzer = TextureQualityAnalyzer()
    results = analyzer.analyze_image(img, expected_product_color="white_cream")
    
    assert results["texture_quality_status"] == "contaminated"
    assert "WHITE_PRODUCT_TEXTURE_RATIO_LOW" in results["texture_quality_reasons"]
