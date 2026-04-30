import requests
import json
import os
import pytest
import cv2
import numpy as np
from pathlib import Path

API_BASE = "https://localhost:8001/api"
TEST_PRODUCT = "test_hardening_product"

def create_valid_tiny_video(path, ext=".mp4"):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if ext == ".mp4" else cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(str(path), fourcc, 20.0, (720, 720))
    for _ in range(300): # 15 seconds at 20fps
        frame = np.random.randint(0, 255, (720, 720, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

@pytest.fixture
def dummy_video_file():
    video_path = Path("test_video_valid.mp4")
    create_valid_tiny_video(video_path, ".mp4")
    yield video_path
    if video_path.exists():
        try:
            video_path.unlink()
        except:
            pass

@pytest.fixture
def dummy_webm_file():
    video_path = Path("test_video_valid.webm")
    create_valid_tiny_video(video_path, ".webm")
    yield video_path
    if video_path.exists():
        try:
            video_path.unlink()
        except:
            pass

def test_reject_missing_manifest(dummy_video_file):
    with open(dummy_video_file, 'rb') as f:
        files = {'file': ('test.mp4', f, 'video/mp4')}
        data = {'product_id': TEST_PRODUCT}
        response = requests.post(f"{API_BASE}/sessions/upload", files=files, data=data, verify=False)
    
    assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text}"
    assert "Missing quality_manifest" in response.text

def test_reject_demo_in_production(dummy_video_file):
    health = requests.get(f"{API_BASE}/health", verify=False).json()
    if health['env'] == 'local_dev':
        pytest.skip("Server is in local_dev, demo mode rejection not testable without config change.")

    manifest = {
        "is_demo": True,
        "product_profile": "generic",
        "coverage_summary": {"percent": 100, "maxGap": 0},
        "accepted_frame_count": 150,
        "total_frame_count": 200
    }
    
    with open(dummy_video_file, 'rb') as f:
        files = {'file': ('test.mp4', f, 'video/mp4')}
        data = {
            'product_id': TEST_PRODUCT,
            'quality_manifest': json.dumps(manifest)
        }
        response = requests.post(f"{API_BASE}/sessions/upload", files=files, data=data, verify=False)
    
    assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text}"
    assert "Demo mode is only permitted in LOCAL_DEV" in response.text

def test_reject_bad_coverage(dummy_video_file):
    manifest = {
        "is_demo": False,
        "product_profile": "generic",
        "coverage_summary": {"percent": 50, "maxGap": 180},
        "accepted_frame_count": 150,
        "total_frame_count": 200,
        "rejection_stats": {}
    }
    
    with open(dummy_video_file, 'rb') as f:
        files = {'file': ('test.mp4', f, 'video/mp4')}
        data = {
            'product_id': TEST_PRODUCT,
            'quality_manifest': json.dumps(manifest)
        }
        response = requests.post(f"{API_BASE}/sessions/upload", files=files, data=data, verify=False)
    
    assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text}"
    result = response.json()
    assert result['detail']['manifest_validation_status'] == "rejected"
    assert any("Coverage too low" in r for r in result['detail']['reasons'])

def test_reject_bad_gap(dummy_video_file):
    manifest = {
        "is_demo": False,
        "product_profile": "generic",
        "coverage_summary": {"percent": 95, "maxGap": 120},
        "accepted_frame_count": 150,
        "total_frame_count": 200,
        "rejection_stats": {}
    }
    
    with open(dummy_video_file, 'rb') as f:
        files = {'file': ('test.mp4', f, 'video/mp4')}
        data = {
            'product_id': TEST_PRODUCT,
            'quality_manifest': json.dumps(manifest)
        }
        response = requests.post(f"{API_BASE}/sessions/upload", files=files, data=data, verify=False)
    
    assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text}"
    result = response.json()
    assert any("Gap too large" in r for r in result['detail']['reasons'])

def test_reject_low_frame_count(dummy_video_file):
    manifest = {
        "is_demo": False,
        "product_profile": "generic",
        "coverage_summary": {"percent": 95, "maxGap": 20},
        "accepted_frame_count": 30,
        "total_frame_count": 100,
        "rejection_stats": {}
    }
    
    with open(dummy_video_file, 'rb') as f:
        files = {'file': ('test.mp4', f, 'video/mp4')}
        data = {
            'product_id': TEST_PRODUCT,
            'quality_manifest': json.dumps(manifest)
        }
        response = requests.post(f"{API_BASE}/sessions/upload", files=files, data=data, verify=False)
    
    assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text}"
    result = response.json()
    assert any("Insufficient accepted frames" in r for r in result['detail']['reasons'])

def test_reject_too_much_blur(dummy_video_file):
    manifest = {
        "is_demo": False,
        "product_profile": "generic",
        "coverage_summary": {"percent": 95, "maxGap": 20},
        "accepted_frame_count": 150,
        "total_frame_count": 300,
        "rejection_stats": {"blur": 150} # 50% blur
    }
    
    with open(dummy_video_file, 'rb') as f:
        files = {'file': ('test.mp4', f, 'video/mp4')}
        data = {
            'product_id': TEST_PRODUCT,
            'quality_manifest': json.dumps(manifest)
        }
        response = requests.post(f"{API_BASE}/sessions/upload", files=files, data=data, verify=False)
    
    assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text}"
    result = response.json()
    assert any("Too much blur" in r for r in result['detail']['reasons'])

def test_accept_webm_ar_blob(dummy_webm_file):
    manifest = {
        "is_demo": False,
        "product_profile": "generic",
        "coverage_summary": {"percent": 95, "maxGap": 20},
        "accepted_frame_count": 150,
        "total_frame_count": 200,
        "rejection_stats": {}
    }
    
    with open(dummy_webm_file, 'rb') as f:
        files = {'file': ('capture_test.webm', f, 'video/webm')}
        data = {
            'product_id': TEST_PRODUCT,
            'quality_manifest': json.dumps(manifest)
        }
        response = requests.post(f"{API_BASE}/sessions/upload", files=files, data=data, verify=False)
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    result = response.json()
    assert result['manifest_validation_status'] == "passed"

def test_accept_valid_manifest(dummy_video_file):
    manifest = {
        "is_demo": False,
        "product_profile": "generic",
        "coverage_summary": {"percent": 95, "maxGap": 20},
        "accepted_frame_count": 150,
        "total_frame_count": 200,
        "rejection_stats": {"blur": 10}
    }
    
    with open(dummy_video_file, 'rb') as f:
        files = {'file': ('test.mp4', f, 'video/mp4')}
        data = {
            'product_id': TEST_PRODUCT,
            'quality_manifest': json.dumps(manifest)
        }
        response = requests.post(f"{API_BASE}/sessions/upload", files=files, data=data, verify=False)
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    result = response.json()
    assert result['manifest_validation_status'] == "passed"

if __name__ == "__main__":
    pytest.main([__file__])
