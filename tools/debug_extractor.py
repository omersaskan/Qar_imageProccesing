import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

# Add modules to path
modules_path = str(Path(__file__).parent.parent.parent)
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

from capture_workflow.frame_extractor import FrameExtractor

@patch('cv2.VideoCapture')
@patch('cv2.imwrite')
def test_direct_extractor(mock_imwrite, mock_videocapture):
    mock_cap = MagicMock()
    mock_videocapture.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    
    frame1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
    mock_cap.read.side_effect = [
        (True, frame1),
        (False, None)
    ]
    
    extractor = FrameExtractor()
    extractor.thresholds.frame_sample_rate = 1
    
    output_dir = "temp_frames"
    keyframes = extractor.extract_keyframes("dummy.mp4", output_dir)
    print(f"Extracted keyframes: {keyframes}")
    assert len(keyframes) == 1

if __name__ == "__main__":
    try:
        test_direct_extractor()
        print("Test Passed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
