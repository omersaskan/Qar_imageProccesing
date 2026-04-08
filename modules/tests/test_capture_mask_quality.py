import pytest
import cv2
import numpy as np
from modules.capture_workflow.object_masker import ObjectMasker
from modules.capture_workflow.quality_analyzer import QualityAnalyzer

def test_mask_metrics_on_synthetic():
    masker = ObjectMasker()
    analyzer = QualityAnalyzer()
    
    # Create frame with a clear object and some "content" to pass exposure/blur
    frame = np.random.randint(20, 40, (400, 400, 3), dtype=np.uint8) # Noisy background
    cv2.circle(frame, (200, 200), 100, (200, 200, 200), -1)
    # Add some texture to circle
    for _ in range(100):
        cv2.circle(frame, (np.random.randint(150, 250), np.random.randint(150, 250)), 2, (100, 100, 100), -1)
    
    mask, meta = masker.generate_mask(frame)
    analysis = analyzer.analyze_frame(frame, mask)
    
    # Assertions
    assert meta["confidence"] > 0.8
    assert meta["solidity"] > 0.95 # Sphere is very solid
    assert meta["fragment_count"] == 1
    assert analysis["overall_pass"] == True
    assert analysis["mask_confidence"] > 0.8

def test_fragmented_mask_rejection():
    masker = ObjectMasker()
    analyzer = QualityAnalyzer()
    
    # Create frame with fragmented "islands"
    frame = np.zeros((800, 800, 3), dtype=np.uint8)
    # Draw larger, distinct dots far apart
    for i in range(10):
        cv2.circle(frame, (80 * i + 40, (i % 3) * 200 + 100), 10, (200, 200, 200), -1)
    
    mask, meta = masker.generate_mask(frame)
    analysis = analyzer.analyze_frame(frame, mask)
    
    # Fragmentation should reduce confidence
    assert meta["fragment_count"] > 1
    if meta["fragment_count"] > 5:
        assert analysis["mask_confidence"] < 0.8

def test_clipping_detection():
    masker = ObjectMasker()
    analyzer = QualityAnalyzer()
    
    # Create frame with object touching edge
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(frame, (0, 200), 80, (200, 200, 200), -1)
    
    mask, meta = masker.generate_mask(frame)
    analysis = analyzer.analyze_frame(frame, mask)
    
    assert meta["is_clipped"] == True
    assert analysis["is_clipped"] == True
    assert "Subject clipped at edges" in analysis["failure_reasons"]

if __name__ == "__main__":
    pytest.main([__file__])
