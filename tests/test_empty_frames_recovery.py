import pytest
import os
import shutil
import json
from pathlib import Path
import cv2
import numpy as np
from modules.operations.worker import IngestionWorker
from modules.operations.settings import settings
from modules.shared_contracts.lifecycle import AssetStatus
from modules.shared_contracts.models import CaptureSession
from datetime import datetime, timezone

def create_valid_tiny_video(path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, 20.0, (720, 720))
    for _ in range(300): # 15 seconds at 20fps
        frame = np.random.randint(0, 255, (720, 720, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

@pytest.fixture
def test_env():
    # Setup a temporary data root
    temp_root = Path("tmp_test_recovery")
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True)
    
    # Save original settings
    original_data_root = settings.data_root
    settings.data_root = str(temp_root)
    
    yield temp_root
    
    # Restore settings and cleanup
    settings.data_root = original_data_root
    if temp_root.exists():
        shutil.rmtree(temp_root)

def test_empty_frames_trigger_extraction(test_env):
    session_id = "cap_test_recovery"
    product_id = "test_product"
    
    # 1. Setup session directories
    capture_path = test_env / "captures" / session_id
    video_dir = capture_path / "video"
    frames_dir = capture_path / "frames"
    video_dir.mkdir(parents=True)
    frames_dir.mkdir(parents=True) # Existing empty frames directory
    
    # 2. Create raw video
    video_path = video_dir / "raw_video.mp4"
    create_valid_tiny_video(video_path)
    
    # 3. Create session JSON
    sessions_dir = test_env / "sessions"
    sessions_dir.mkdir(parents=True)
    session_file = sessions_dir / f"{session_id}.json"
    
    session_data = {
        "session_id": session_id,
        "product_id": product_id,
        "status": AssetStatus.CAPTURED.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "retry_count": 0,
        "operator_id": "test_user"
    }
    
    with open(session_file, "w") as f:
        json.dump(session_data, f)
        
    worker = IngestionWorker(data_root=str(test_env))
    session = CaptureSession.model_validate(session_data)
    
    # We want to test that _handle_reconstruction calls _handle_frame_extraction
    # if frames are missing.
    # To avoid running full reconstruction (which requires COLMAP), 
    # we can mock the reconstruction logic or just catch the error AFTER extraction.
    
    from unittest.mock import patch
    
    # Check that frames dir is empty initially
    assert len(list(frames_dir.glob("*.jpg"))) == 0
    
    # Mock FrameExtractor to actually produce some files
    dummy_frame = frames_dir / "frame_0001.jpg"
    
    with patch("modules.capture_workflow.frame_extractor.FrameExtractor.extract_keyframes") as mock_extract:
        # Side effect to create multiple dummy files so we pass the min_frames check (5)
        def side_effect(*args, **kwargs):
            frames = []
            for i in range(5):
                f = frames_dir / f"frame_{i:04d}.jpg"
                f.touch()
                frames.append(str(f))
            return frames, {"status": "ok"}
            
        mock_extract.side_effect = side_effect
        
        try:
            # This will trigger _handle_frame_extraction first
            worker._handle_reconstruction(session)
        except Exception as e:
            # It might fail later due to missing COLMAP, but we care about the frames
            print(f"Caught expected later failure: {e}")
    
    # 4. Verify frames were extracted (via our mock)
    extracted_frames = list(frames_dir.glob("*.jpg"))
    assert len(extracted_frames) > 0, "Frames should have been extracted because the directory was empty."

    
    # 5. Verify session was updated with extracted frames
    updated_session = worker.session_manager.get_session(session_id)
    assert updated_session.extracted_frames is not None
    assert len(updated_session.extracted_frames) > 0
