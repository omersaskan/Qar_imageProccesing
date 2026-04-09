import pytest
import numpy as np
import sys
from unittest.mock import patch, MagicMock

from modules.capture_workflow.config import SegmentationConfig
from modules.capture_workflow.segmentation_backends.factory import BackendFactory
from modules.capture_workflow.object_masker import ObjectMasker
from modules.capture_workflow.quality_analyzer import QualityAnalyzer

@pytest.fixture
def synthetic_frame():
    # 100x100 white image
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    # Add a black square in the middle to simulate an object
    frame[30:70, 30:70] = 0
    return frame

def test_rembg_happy_path(synthetic_frame):
    config = SegmentationConfig(backend="rembg", fallback_backend="heuristic", hard_fail_on_backend_error=False)
    
    mock_rembg = MagicMock()
    mock_out = np.zeros((100, 100, 4), dtype=np.uint8)
    mock_out[30:70, 30:70, 3] = 255 # absolute white mask
    mock_rembg.remove.return_value = mock_out

    with patch.dict(sys.modules, {"rembg": mock_rembg}):
        masker = ObjectMasker(config=config)
        mask, meta = masker.generate_mask(synthetic_frame)

        assert mock_rembg.remove.called
        assert meta["backend_name"] == "rembg"
        assert not meta["fallback_used"]
        # It's an array of shape (100, 100)
        assert mask.shape == (100, 100)

def test_rembg_failure_fallback(synthetic_frame):
    config = SegmentationConfig(backend="rembg", fallback_backend="heuristic", hard_fail_on_backend_error=False)

    mock_rembg = MagicMock()
    mock_rembg.remove.side_effect = Exception("Model failed to load")

    with patch.dict(sys.modules, {"rembg": mock_rembg}):
        masker = ObjectMasker(config=config)
        mask, meta = masker.generate_mask(synthetic_frame)
        
        assert meta["fallback_used"] is True
        assert meta["fallback_reason"] == "Model failed to load"
        # Since fallback is heuristic, it should provide a mask
        assert mask.shape == (100, 100)

def test_metadata_correctness(synthetic_frame):
    config = SegmentationConfig(backend="heuristic")
    masker = ObjectMasker(config=config)
    mask, meta = masker.generate_mask(synthetic_frame)

    assert meta["backend_name"] == "heuristic"
    assert "mask_confidence" in meta
    assert "purity_score" in meta
    assert "support_suspected" in meta
    assert "fallback_used" in meta

def test_quality_analyzer_semantic_handling():
    analyzer = QualityAnalyzer()
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.full((100, 100), 0, dtype=np.uint8)
    mask[30:70, 30:70] = 255 # valid occupancy

    meta = {
        "bbox": {"x": 30, "y": 30, "w": 40, "h": 40},
        "centroid": {"x": 50.0, "y": 50.0},
        "fragment_count": 1,
        "largest_contour_ratio": 1.0,
        "solidity": 1.0,
        "mask_confidence": 0.4, # Too low! (default threshold is 0.55)
        "purity_score": 0.9,
        "fallback_used": True,
        "backend_name": "heuristic",
    }

    result = analyzer.analyze_frame(frame, mask, meta)
    assert not result["overall_pass"]
    assert "fallback_mask_used_with_low_confidence (0.40)" in result["failure_reasons"]
    assert "low_mask_confidence (0.40)" in result["failure_reasons"]
