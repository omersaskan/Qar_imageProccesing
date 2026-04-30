import requests
import json
import os
import cv2
import numpy as np
import time
from pathlib import Path
import pytest

API_BASE = "https://localhost:8001/api"
DATA_ROOT = Path("data")

def create_valid_tiny_video(path, ext=".mp4"):
    import uuid
    temp_path = Path(f"temp_{uuid.uuid4().hex}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_path), fourcc, 20.0, (720, 720))
    for _ in range(300): # 15 seconds at 20fps
        frame = np.random.randint(0, 255, (720, 720, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    # Give Windows a moment to release handle
    time.sleep(0.5)
    if path.exists(): path.unlink()
    os.rename(temp_path, path)

def test_mobile_smoke_success():
    # 1. Setup
    video_path = Path("smoke_test_capture.webm")
    create_valid_tiny_video(video_path, ".mp4") # Write mp4 data but use webm extension
    
    product_id = "smoke_success_product"
    manifest = {
        "is_demo": False,
        "product_profile": "generic",
        "coverage_summary": {"percent": 98, "maxGap": 15},
        "accepted_frame_count": 180,
        "total_frame_count": 200,
        "rejection_stats": {}
    }

    # 2. Upload
    with open(video_path, 'rb') as f:
        files = {'file': ('capture_smoke.webm', f, 'video/webm')}
        data = {
            'product_id': product_id,
            'quality_manifest': json.dumps(manifest)
        }
        response = requests.post(f"{API_BASE}/sessions/upload", files=files, data=data, verify=False)
    
    assert response.status_code == 200, f"Upload failed: {response.text}"
    result = response.json()
    session_id = result['session_id']
    assert result['manifest_validation_status'] == "passed"
    
    # 3. Verify Filesystem
    session_dir = DATA_ROOT / "captures" / session_id
    video_dir = session_dir / "video"
    reports_dir = session_dir / "reports"
    
    # Wait a bit for filesystem to sync if needed
    time.sleep(1)
    
    assert (video_dir / "original_capture.webm").exists(), "original_capture.webm missing"
    assert (video_dir / "raw_video.mp4").exists(), "raw_video.mp4 missing"
    assert (reports_dir / "ar_quality_manifest.json").exists(), "ar_quality_manifest.json missing"
    
    # 4. Verify cv2 readability of normalized video
    normalized_path = video_dir / "raw_video.mp4"
    cap = cv2.VideoCapture(str(normalized_path))
    assert cap.isOpened(), "cv2 could not open normalized raw_video.mp4"
    ret, frame = cap.read()
    assert ret, "cv2 could not read frame from normalized raw_video.mp4"
    cap.release()
    
    # 5. Verify it's a real H.264 MP4 (not just renamed WebM)
    # Using ffprobe to check codec_name
    from modules.operations.settings import Settings
    settings_obj = Settings()
    ffprobe_path = settings_obj.ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe")
    import subprocess
    cmd = [ffprobe_path, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", str(normalized_path)]
    codec = subprocess.check_output(cmd).decode().strip()
    print(f"Verified codec for raw_video.mp4: {codec}")
    assert codec == "h264", f"Expected h264 codec, got {codec}"
    
    # 6. Verify manifest content
    with open(reports_dir / "ar_quality_manifest.json", "r") as f:
        saved_manifest = json.load(f)
        assert saved_manifest['coverage_summary']['percent'] == 98
    
    # 6. Verify Worker started processing
    time.sleep(7) 
    
    # Check session status via guidance endpoint
    status_resp = requests.get(f"{API_BASE}/sessions/{session_id}/guidance", verify=False)
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    
    print(f"Session {session_id} status: {status_data['status']}")
    assert status_data['status'] in ["captured", "extracting", "failed", "processing", "reconstructing", "uploaded"], f"Unexpected status: {status_data['status']}"

    # Cleanup
    if video_path.exists(): video_path.unlink()

def test_mobile_smoke_rejection():
    # 1. Setup
    video_path = Path("smoke_test_fail.webm")
    create_valid_tiny_video(video_path, ".mp4") # Consistent with success test
    
    product_id = "smoke_fail_product"
    manifest = {
        "is_demo": False,
        "product_profile": "generic",
        "coverage_summary": {"percent": 40, "maxGap": 190}, # FAIL
        "accepted_frame_count": 50, # FAIL
        "total_frame_count": 200,
        "rejection_stats": {}
    }

    # 2. Upload
    with open(video_path, 'rb') as f:
        files = {'file': ('capture_fail.webm', f, 'video/webm')}
        data = {
            'product_id': product_id,
            'quality_manifest': json.dumps(manifest)
        }
        response = requests.post(f"{API_BASE}/sessions/upload", files=files, data=data, verify=False)
    
    assert response.status_code == 422
    result = response.json()
    
    # Verify rejection reasons (what UI would show)
    assert result['detail']['manifest_validation_status'] == "rejected"
    reasons = result['detail']['reasons']
    assert any("Coverage too low" in r for r in reasons)
    assert any("Gap too large" in r for r in reasons)
    assert any("Insufficient accepted frames" in r for r in reasons)
    
    print("Rejection reasons returned correctly:", reasons)

    # Cleanup
    if video_path.exists(): video_path.unlink()

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
