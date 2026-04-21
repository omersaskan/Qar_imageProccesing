import pytest
import numpy as np
import cv2
import json
from pathlib import Path
from unittest.mock import patch

from modules.shared_contracts.models import CaptureSession, AssetStatus
from modules.capture_workflow.frame_extractor import FrameExtractor
from modules.capture_workflow.coverage_analyzer import CoverageAnalyzer
from modules.capture_workflow.config import ExtractionConfig, CoverageConfig, SegmentationConfig
from modules.operations.worker import IngestionWorker

@pytest.fixture
def test_workspace():
    # Use local relative path to avoid Unicode username paths (e.g. Ömer) which fail cv2.imwrite on Windows
    root = Path("tmp/test_workspace")
    if root.exists():
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    sessions = root / "sessions"
    sessions.mkdir()
    captures = root / "captures"
    captures.mkdir()
    yield root
    import shutil
    shutil.rmtree(root, ignore_errors=True)

@pytest.fixture
def mock_session(test_workspace):
    session = CaptureSession(
        session_id="test_session_1",
        product_id="test_prod",
        operator_id="test_worker_1",
        status=AssetStatus.CREATED,
    )
    session_file = test_workspace / "sessions" / f"{session.session_id}.json"
    cap_dir = test_workspace / "captures" / session.session_id / "video"
    cap_dir.mkdir(parents=True)
    video_path = cap_dir / "raw_video.mp4"
    video_path.touch()
    
    with open(session_file, "w") as f:
        f.write(session.model_dump_json())
        
    return session

def test_frame_extractor_integration(test_workspace, mock_session):
    video_path = test_workspace / "captures" / mock_session.session_id / "video" / "raw_video.mp4"
    frames_dir = test_workspace / "captures" / mock_session.session_id / "frames"
    
    config = ExtractionConfig(min_frames=1, max_frames=2)
    # Fast heuristic override for testing logic path
    seg_config = SegmentationConfig(backend="heuristic")
    
    extractor = FrameExtractor(config=config, seg_config=seg_config)

    # Use cv2.VideoCapture mock to yield synthetic frames to test extraction logic
    cap_mock = patch("cv2.VideoCapture")
    with cap_mock as mock:
        instance = mock.return_value
        instance.isOpened.return_value = True
        
        # Generator for frames, needs to be large enough to pass min_object_occupancy defaults
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        frame[35:65, 35:65] = 0 # 30x30 object = 900 area = 0.09 occupancy (default min is 0.05)
        responses = [(True, frame), (True, frame), (True, frame), (False, None)]
        def ret_logic():
            if responses:
                return responses.pop(0)
            return (False, None)
        instance.read.side_effect = ret_logic

        # We expect 1 frame since the generator yields similar mock frames 
        # and deduplication drops the rest, except the first acceptable.
        frames, _report = extractor.extract_keyframes(str(video_path), str(frames_dir))
        assert len(frames) == 1
        assert Path(frames[0]).name == "frame_0000.jpg"
        
        mask_path = frames_dir / "masks" / "frame_0000.png"
        meta_path = frames_dir / "masks" / "frame_0000.json"
        assert mask_path.exists(), "Mask png was not saved alongside the frame"
        assert meta_path.exists(), "Metadata json was not saved alongside the frame"
        
        with open(meta_path, "r") as f:
            meta = json.load(f)
            assert meta["backend_name"] == "heuristic", "Fallback config failed to apply"
            assert "mask_confidence" in meta, "Fallback semantic metrics missing"

def test_worker_recapture_path_regression(test_workspace, mock_session):
    worker = IngestionWorker(data_root=str(test_workspace))
    worker.session_manager.sessions_dir = test_workspace / "sessions"
    worker.session_manager.captures_dir = test_workspace / "captures"
    
    from modules.operations.worker import IrrecoverableError

    # Mock extract_keyframes to return empty or low diversity to force recapture
    with patch("modules.capture_workflow.frame_extractor.FrameExtractor.extract_keyframes") as mock_ext:
        mock_ext.return_value = ([], {})
        
        with pytest.raises(IrrecoverableError, match=r"produced 0 frames"):
            worker._handle_frame_extraction(mock_session)
            
        mock_ext.return_value = (["frame1.jpg", "frame2.jpg", "frame3.jpg", "frame4.jpg", "frame5.jpg"], {})
        with patch("modules.capture_workflow.coverage_analyzer.CoverageAnalyzer.analyze_coverage") as mock_cov:
            mock_cov.return_value = {
                "overall_status": "insufficient",
                "coverage_score": 0.2,
                "reasons": ["fallback used too much"],
                "hard_reasons": ["fallback used too much"]
            }
            res = worker._handle_frame_extraction(mock_session)
            assert res.status == AssetStatus.RECAPTURE_REQUIRED
            assert res.publish_state == "needs_recapture"
            assert "fallback used too much" in res.failure_reason
