import pytest
import cv2
import numpy as np
from unittest.mock import MagicMock, patch
from modules.capture_workflow.config import QualityThresholds
from modules.capture_workflow.frame_extractor import FrameExtractor


def _make_test_frame() -> np.ndarray:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = 40
    frame[:, :, 1] = 80
    frame[:, :, 2] = 120
    frame[20:80, 20:80, :] = 220
    return frame


@patch('cv2.VideoCapture')
def test_extract_keyframes_mocked(mock_videocapture, tmp_path):
    mock_cap = MagicMock()
    mock_videocapture.return_value = mock_cap
    mock_cap.isOpened.return_value = True

    frame1 = _make_test_frame()
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
        thresholds=QualityThresholds(frame_sample_rate=1),
    )

    output_dir = tmp_path / "frames"
    with patch.object(extractor, "_write_verified_image") as mock_write_verified_image:
        frames, _report = extractor.extract_keyframes("dummy.mp4", str(output_dir))

    assert len(frames) == 1
    assert mock_write_verified_image.call_count == 3
    assert _report["frame_mode"] == "raw_for_reconstruction"
    assert "masked_preview_dir" in _report
    assert "masks_dir" in _report

@patch('cv2.VideoCapture')
@patch('cv2.imencode')
def test_extract_keyframes_hard_fails_on_write_error(mock_imencode, mock_videocapture, tmp_path):
    mock_cap = MagicMock()
    mock_videocapture.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [
        (True, _make_test_frame()),
        (False, None),
    ]
    # Simulate cv2.imencode failure: success=False
    mock_imencode.return_value = (False, None)

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
        thresholds=QualityThresholds(frame_sample_rate=1),
    )

    with pytest.raises(ValueError, match="write failed|internal encode failed"):
        extractor.extract_keyframes("dummy.mp4", str(tmp_path / "frames"))

def test_histogram_generation():
    extractor = FrameExtractor()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    hist = extractor._get_masked_histogram(frame, mask)
    assert hist is not None
    assert hist.shape == (180, 128)


def test_roi_preparation_preserves_canvas_shape():
    extractor = FrameExtractor()
    frame = _make_test_frame()
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255

    focused_frame, focused_mask = extractor._prepare_object_centric_frame(
        frame,
        mask,
        {"x": 20, "y": 20, "w": 60, "h": 60},
    )

    assert focused_frame.shape == frame.shape
    assert focused_mask.shape == mask.shape
