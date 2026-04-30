import pytest
import os
import json
import subprocess
from pathlib import Path
from fastapi.testclient import TestClient
from modules.operations.api import app
from modules.operations.settings import settings

client = TestClient(app)

@pytest.fixture
def test_video(tmp_path):
    """Creates a real WebM test video using ffmpeg."""
    video_path = tmp_path / "test.webm"
    cmd = [
        settings.ffmpeg_path or "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=5:size=1280x720:rate=30",
        "-c:v", "libvpx-vp9",
        "-f", "webm",
        str(video_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return video_path

def test_upload_reordered_flow(test_video):
    """Verifies that WebM is accepted and normalized before validation."""
    # 1. Valid manifest
    manifest = {
        "total_frame_count": 150,
        "accepted_frame_count": 140,
        "blur_rejection_count": 5,
        "coverage_summary": {"percent": 95, "maxGap": 10},
        "profile_type": "generic",
        "is_demo": False
    }
    
    with open(test_video, "rb") as f:
        response = client.post(
            "/api/sessions/upload",
            data={
                "product_id": "test_prod",
                "quality_manifest": json.dumps(manifest)
            },
            files={"file": ("test.webm", f, "video/webm")},
            headers={"X-API-Key": settings.pilot_api_key or "test_key"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    
    # Verify file was normalized
    session_id = data["session_id"]
    normalized_path = Path(settings.data_root) / "captures" / session_id / "video" / "raw_video.mp4"
    assert normalized_path.exists()

def test_upload_manifest_json_error(test_video):
    """Verifies 400 error for malformed JSON manifest."""
    with open(test_video, "rb") as f:
        response = client.post(
            "/api/sessions/upload",
            data={
                "product_id": "test_prod",
                "quality_manifest": "{invalid_json"
            },
            files={"file": ("test.webm", f, "video/webm")},
            headers={"X-API-Key": settings.pilot_api_key or "test_key"}
        )
    
    assert response.status_code == 400
    assert "Malformed JSON" in response.json()["detail"]

def test_upload_ar_quality_gate_failure(test_video):
    """Verifies 422 error for quality gate failure (e.g. high blur)."""
    manifest = {
        "total_frame_count": 150,
        "accepted_frame_count": 140,
        "blur_rejection_count": 100, # High blur ratio
        "coverage_summary": {"percent": 95, "maxGap": 10},
        "profile_type": "generic"
    }
    
    with open(test_video, "rb") as f:
        response = client.post(
            "/api/sessions/upload",
            data={
                "product_id": "test_prod",
                "quality_manifest": json.dumps(manifest)
            },
            files={"file": ("test.webm", f, "video/webm")},
            headers={"X-API-Key": settings.pilot_api_key or "test_key"}
        )
    
    assert response.status_code == 422
    assert "Blur ratio too high" in str(response.json()["detail"])

def test_upload_box_profile_gate(test_video):
    """Verifies box profile requires 6 faces."""
    manifest = {
        "total_frame_count": 150,
        "accepted_frame_count": 140,
        "blur_rejection_count": 5,
        "coverage_summary": {"percent": 95, "maxGap": 10},
        "profile_type": "box",
        "profile_completion": {"faces": ["front", "back"]} # Only 2 faces
    }
    
    with open(test_video, "rb") as f:
        response = client.post(
            "/api/sessions/upload",
            data={
                "product_id": "test_prod",
                "quality_manifest": json.dumps(manifest)
            },
            files={"file": ("test.webm", f, "video/webm")},
            headers={"X-API-Key": settings.pilot_api_key or "test_key"}
        )
    
    assert response.status_code == 422
    assert "Box requires 6 completed faces" in str(response.json()["detail"])
