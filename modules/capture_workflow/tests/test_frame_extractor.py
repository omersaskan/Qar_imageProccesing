import pytest
import cv2
import numpy as np
from unittest.mock import MagicMock, patch
from modules.capture_workflow.frame_extractor import FrameExtractor

@patch('cv2.VideoCapture')
@patch('cv2.imwrite')
def test_extract_keyframes_mocked(mock_imwrite, mock_videocapture, tmp_path):
    mock_cap = MagicMock()
    mock_videocapture.return_value = mock_cap
    mock_cap.isOpened.return_value = True

    frame1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    frame3 = frame1.copy()

    mock_cap.read.side_effect = [
        (True, frame1),
        (True, frame2),
        (True, frame3),
        (False, None)
    ]

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    mask_meta = {"bbox": {"x": 20, "y": 20, "w": 60, "h": 60}}

    object_masker = MagicMock()
    object_masker.generate_mask.return_value = (mask, mask_meta)

    quality_analyzer = MagicMock()
    quality_analyzer.analyze_frame.return_value = {"overall_pass": True, "failure_reasons": []}

    extractor = FrameExtractor(
        quality_analyzer=quality_analyzer,
        object_masker=object_masker,
    )
    extractor.thresholds.frame_sample_rate = 1

    output_dir = tmp_path / "frames"
    keyframes = extractor.extract_keyframes("dummy.mp4", str(output_dir))

    assert len(keyframes) == 1
    assert mock_imwrite.call_count == 2

def test_histogram_generation():
    extractor = FrameExtractor()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    hist = extractor._get_masked_histogram(frame, mask)
    assert hist is not None
    assert hist.shape == (180, 128)
