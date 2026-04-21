import pytest
import os
import json
import shutil
from pathlib import Path
from modules.operations.worker import IngestionWorker
from modules.capture_workflow.session_manager import SessionManager
from modules.shared_contracts.lifecycle import AssetStatus

@pytest.fixture
def clean_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sessions").mkdir()
    (data_dir / "captures").mkdir()
    
    # Use monkeypatch to redirect worker/manager to tmp_path
    monkeypatch.chdir(tmp_path)
    return data_dir

def test_worker_fails_irrecoverable_on_missing_video(clean_data):
    manager = SessionManager(data_root=str(clean_data))
    worker = IngestionWorker()
    
    # 1. Create a session
    session = manager.create_session("sess_fail_1", "prod_1", "op_1")
    
    # 2. Advance to CAPTURED (should fail because no video)
    # The worker _run loop calls _process_pending_sessions
    worker._process_pending_sessions()
    
    # 3. Verify session is now FAILED
    updated_session = manager.get_session("sess_fail_1")
    assert updated_session.status == AssetStatus.FAILED
    assert "Video file missing" in updated_session.failure_reason

def test_worker_fails_on_zero_frames(clean_data, monkeypatch):
    manager = SessionManager(data_root=str(clean_data))
    worker = IngestionWorker()
    
    # 1. Create a session
    session = manager.create_session("sess_zero_frames", "prod_1", "op_1")
    
    # 2. Add an empty video file
    video_path = Path(clean_data) / "captures" / "sess_zero_frames" / "video" / "raw_video.mp4"
    video_path.touch()
    
    # 3. Mock FrameExtractor to return 0 frames
    class MockExtractor:
        def extract_keyframes(self, *args, **kwargs): return ([], {})
        
    monkeypatch.setattr("modules.capture_workflow.frame_extractor.FrameExtractor", MockExtractor)
    
    # 4. Run worker
    worker._process_pending_sessions()
    
    # 5. Verify failed
    updated_session = manager.get_session("sess_zero_frames")
    assert updated_session.status == AssetStatus.FAILED
    assert "Frame extraction produced 0 frames" in updated_session.failure_reason
