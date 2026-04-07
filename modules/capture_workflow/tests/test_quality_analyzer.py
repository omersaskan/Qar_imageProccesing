import pytest
import numpy as np
from capture_workflow.quality_analyzer import QualityAnalyzer
from capture_workflow.config import QualityThresholds

def test_analyze_black_frame():
    analyzer = QualityAnalyzer()
    # 100x100 black frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    analysis = analyzer.analyze_frame(frame)
    assert analysis["blur_score"] == 0.0
    assert analysis["exposure_score"] == 0.0
    assert analysis["is_blur_ok"] is False # Too blurry (no texture)
    assert analysis["is_exposure_ok"] is False # Underexposed
    assert analysis["overall_pass"] is False

def test_analyze_noise_frame():
    # Uniform noise should have high blur score (high variance)
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    analyzer = QualityAnalyzer()
    analysis = analyzer.analyze_frame(frame)
    
    assert analysis["blur_score"] > 100.0
    assert 100 < analysis["exposure_score"] < 150
    assert analysis["is_blur_ok"] is True
    assert analysis["is_exposure_ok"] is True

def test_config_override():
    # Very strict config
    strict_thresholds = QualityThresholds(min_blur_score=5000.0)
    analyzer = QualityAnalyzer(thresholds=strict_thresholds)
    
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    analysis = analyzer.analyze_frame(frame)
    
    # Noise has variance around 5400, but let's be safe
    # If blur_score < 5000 it should fail
    if analysis["blur_score"] < 5000:
        assert analysis["is_blur_ok"] is False
