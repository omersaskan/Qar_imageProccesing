import pytest
import os
import json
from pathlib import Path
from modules.operations.worker import IngestionWorker
from modules.shared_contracts.models import CaptureSession
from modules.shared_contracts.lifecycle import AssetStatus

def test_worker_irrecoverable_failure_video_missing(tmp_path, monkeypatch):
    # Setup worker and paths
    # We'll use monkeypatch to point the worker to a temp directory
    sessions_dir = tmp_path / "sessions"
    captures_dir = tmp_path / "captures"
    sessions_dir.mkdir(parents=True)
    captures_dir.mkdir(parents=True)
    
    monkeypatch.setattr("modules.operations.worker.Path", lambda p: tmp_path / p if str(p).startswith("data") else Path(p))
    
    worker = IngestionWorker()
    worker.session_manager.sessions_dir = sessions_dir
    worker.session_manager.captures_dir = captures_dir
    
    # 1. Create a session that is CREATED
    session_id = "test_fail_sess"
    session = CaptureSession(
        session_id=session_id,
        product_id="test_prod",
        operator_id="test_ops",
        status=AssetStatus.CREATED
    )
    
    session_file = sessions_dir / f"{session_id}.json"
    session_file.write_text(session.model_dump_json())
    
    # Ensure capture dir exists but video is MISSING
    sess_capture_dir = captures_dir / session_id / "video"
    sess_capture_dir.mkdir(parents=True)
    
    # 2. Run worker processing
    worker._process_pending_sessions()
    
    # 3. Verify session is now FAILED
    with open(session_file, "r") as f:
        updated_data = json.load(f)
        updated_session = CaptureSession.model_validate(updated_data)
        
    assert updated_session.status == AssetStatus.FAILED
    assert "Video file missing" in updated_session.failure_reason

def test_worker_irrecoverable_failure_zero_frames(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    captures_dir = tmp_path / "captures"
    sessions_dir.mkdir(parents=True)
    captures_dir.mkdir(parents=True)
    
    worker = IngestionWorker()
    worker.session_manager.sessions_dir = sessions_dir
    worker.session_manager.captures_dir = captures_dir
    
    # Mock FrameExtractor to return 0 frames
    class MockExtractor:
        def extract_keyframes(self, *args, **kwargs):
            return []
            
    monkeypatch.setattr("modules.capture_workflow.frame_extractor.FrameExtractor", MockExtractor)
    
    # 1. Create a session that is CREATED
    session_id = "test_fail_frames"
    session = CaptureSession(
        session_id=session_id,
        product_id="test_prod",
        operator_id="test_ops",
        status=AssetStatus.CREATED
    )
    
    session_file = sessions_dir / f"{session_id}.json"
    session_file.write_text(session.model_dump_json())
    
    # Ensure video file "exists" dummy
    sess_video_dir = captures_dir / session_id / "video"
    sess_video_dir.mkdir(parents=True)
    (sess_video_dir / "raw_video.mp4").write_text("dummy video")
    
    # 2. Run worker processing
    worker._process_pending_sessions()
    
    # 3. Verify session is now FAILED
    with open(session_file, "r") as f:
        updated_data = json.load(f)
        updated_session = CaptureSession.model_validate(updated_data)
        
    assert updated_session.status == AssetStatus.FAILED
    assert "0 frames" in updated_session.failure_reason


def test_worker_irrecoverable_failure_insufficient_coverage(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    captures_dir = tmp_path / "captures"
    sessions_dir.mkdir(parents=True)
    captures_dir.mkdir(parents=True)

    worker = IngestionWorker()
    worker.session_manager.sessions_dir = sessions_dir
    worker.session_manager.captures_dir = captures_dir

    class MockExtractor:
        def __init__(self):
            self.config = type("Config", (), {"min_frames": 3})()

        def extract_keyframes(self, *args, **kwargs):
            return [f"frame_{idx}.jpg" for idx in range(5)]

    class MockCoverageAnalyzer:
        def analyze_coverage(self, frames):
            return {
                "overall_status": "insufficient",
                "coverage_score": 0.22,
                "reasons": ["Insufficient viewpoint diversity (2/6 unique views)."],
            }

    monkeypatch.setattr("modules.capture_workflow.frame_extractor.FrameExtractor", MockExtractor)
    monkeypatch.setattr("modules.capture_workflow.coverage_analyzer.CoverageAnalyzer", MockCoverageAnalyzer)

    session_id = "test_fail_coverage"
    session = CaptureSession(
        session_id=session_id,
        product_id="test_prod",
        operator_id="test_ops",
        status=AssetStatus.CREATED
    )

    session_file = sessions_dir / f"{session_id}.json"
    session_file.write_text(session.model_dump_json())

    sess_video_dir = captures_dir / session_id / "video"
    sess_video_dir.mkdir(parents=True)
    (sess_video_dir / "raw_video.mp4").write_text("dummy video")

    worker._process_pending_sessions()

    with open(session_file, "r") as f:
        updated_data = json.load(f)
        updated_session = CaptureSession.model_validate(updated_data)

    assert updated_session.status == AssetStatus.RECAPTURE_REQUIRED
    assert updated_session.publish_state == "needs_recapture"
    assert "Capture quality insufficient" in updated_session.failure_reason
