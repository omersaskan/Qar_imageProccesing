import pytest
import os
import cv2
import numpy as np
from unittest.mock import MagicMock, patch
from capture_workflow.frame_extractor import FrameExtractor

@patch('cv2.VideoCapture')
@patch('cv2.imwrite')
def test_extract_keyframes_mocked(mock_imwrite, mock_videocapture, tmp_path):
    # Setup mock video capture
    mock_cap = MagicMock()
    mock_videocapture.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    
    # 3 frames: 1st is good, 2nd is skipped (sample rate), 3rd is same as 1st (similarity)
    # Use noise to pass the blur filter, and identical frames to test similarity
    frame1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    frame3 = frame1.copy()
    
    # Return 3 frames then stop
    mock_cap.read.side_effect = [
        (True, frame1),
        (True, frame2),
        (True, frame3),
        (False, None)
    ]
    
    extractor = FrameExtractor()
    # Force sample rate to 1 to check similarity
    extractor.thresholds.frame_sample_rate = 1
    
    output_dir = tmp_path / "frames"
    keyframes = extractor.extract_keyframes("dummy.mp4", str(output_dir))
    
    # Only 1 keyframe should be extracted because 2 and 3 are same as 1
    assert len(keyframes) == 1
    assert mock_imwrite.called

def test_histogram_generation():
    extractor = FrameExtractor()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hist = extractor._get_histogram(frame)
    assert hist is not None
    assert hist.shape == (180, 256)
