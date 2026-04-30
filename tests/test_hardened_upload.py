import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json
import shutil
import cv2
import numpy as np
import os
from modules.operations.api import app
from modules.operations.settings import settings
from modules.shared_contracts.lifecycle import AssetStatus

client = TestClient(app)

@pytest.fixture
def test_data_root(tmp_path):
    orig_root = settings.data_root
    settings.data_root = str(tmp_path)
    orig_duration = settings.min_video_duration_sec
    settings.min_video_duration_sec = 15.0
    
    # Re-init managers if they were already created in api.py
    # But api.py uses settings.data_root at import time for registry/session_manager
    # So we might need to patch them.
    from modules.operations import api
    from modules.capture_workflow.session_manager import SessionManager
    from modules.asset_registry.registry import AssetRegistry
    
    api.registry = AssetRegistry(data_root=str(tmp_path / "registry"))
    api.session_manager = SessionManager(data_root=str(tmp_path))
    
    yield tmp_path
    settings.data_root = orig_root
    settings.min_video_duration_sec = orig_duration

def create_dummy_video(path: Path, fps=30, duration=2, width=1280, height=720):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for _ in range(int(fps * duration)):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

def test_upload_success(test_data_root):
    video_path = test_data_root / "test.mp4"
    create_dummy_video(video_path, fps=30, duration=settings.min_video_duration_sec + 1)
    
    manifest = {
        "total_frame_count": 150,
        "accepted_frame_count": 120,
        "coverage_summary": {"percent": 95, "maxGap": 10},
        "product_profile": "generic",
        "is_demo": True # In local_dev this is allowed
    }
    
    with open(video_path, "rb") as f:
        response = client.post(
            "/api/sessions/upload",
            data={
                "product_id": "test_prod",
                "operator_id": "test_op",
                "quality_manifest": json.dumps(manifest)
            },
            files={"file": ("test.mp4", f, "video/mp4")}
        )
    
    if response.status_code != 200:
        print(f"FAILED Response: {response.status_code} {response.json()}")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["status"] == "uploaded"
    
    # Check if files exist
    session_id = data["session_id"]
    capture_dir = test_data_root / "captures" / session_id
    assert (capture_dir / "video" / "raw_video.mp4").exists()
    assert (capture_dir / "reports" / "ar_quality_manifest.json").exists()

def test_upload_invalid_product_id(test_data_root):
    response = client.post(
        "/api/sessions/upload",
        data={"product_id": "invalid/id", "operator_id": "op"},
        files={"file": ("test.mp4", b"dummy", "video/mp4")}
    )
    assert response.status_code == 400
    assert "Invalid identifier" in response.json()["detail"]

def test_upload_low_coverage_rejection(test_data_root):
    video_path = test_data_root / "test.mp4"
    create_dummy_video(video_path, fps=30, duration=settings.min_video_duration_sec + 1)
    
    manifest = {
        "total_frame_count": 100,
        "accepted_frame_count": 80,
        "coverage_summary": {"percent": 10, "maxGap": 300}, # Too low
        "product_profile": "generic"
    }
    
    with open(video_path, "rb") as f:
        response = client.post(
            "/api/sessions/upload",
            data={
                "product_id": "test_prod",
                "quality_manifest": json.dumps(manifest)
            },
            files={"file": ("test.mp4", f, "video/mp4")}
        )
    
    assert response.status_code == 422
    assert "reasons" in response.json()["detail"]
    
    # Ensure capture path cleaned up
    # Since session wasn't created, captures dir should be empty or non-existent for this ID
    # But we don't know the ID yet. Let's check the captures root.
    captures_dir = test_data_root / "captures"
    if captures_dir.exists():
        assert len(list(captures_dir.iterdir())) == 0

def test_upload_invalid_video_duration(test_data_root):
    video_path = test_data_root / "too_short.mp4"
    create_dummy_video(video_path, fps=30, duration=1) # 1 sec is too short (min 15 in user req, but let's see settings)
    
    manifest = {
        "total_frame_count": 30,
        "accepted_frame_count": 25,
        "coverage_summary": {"percent": 90, "maxGap": 20}
    }
    
    with open(video_path, "rb") as f:
        response = client.post(
            "/api/sessions/upload",
            data={
                "product_id": "test_prod",
                "quality_manifest": json.dumps(manifest)
            },
            files={"file": ("test.mp4", f, "video/mp4")}
        )
    
    assert response.status_code == 400
    assert "validation failed" in response.json()["detail"] or "Duration too short" in response.json()["detail"]

def test_readiness_ffmpeg(test_data_root):
    response = client.get("/api/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ffmpeg_probe_ok" in data["preflight"]
    assert "ffprobe_probe_ok" in data["preflight"]
