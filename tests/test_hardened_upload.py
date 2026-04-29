import requests
import json
import os
import pytest
from pathlib import Path

API_BASE = "http://localhost:8001/api"
TEST_PRODUCT = "test_hardening_product"

def create_dummy_video():
    return b"fake video content that is large enough to pass size checks" * 1000

@pytest.fixture
def dummy_video_file():
    content = create_dummy_video()
    video_path = Path("test_video.mp4")
    with open(video_path, "wb") as f:
        f.write(content)
    yield video_path
    if video_path.exists():
        video_path.unlink()

def test_reject_missing_manifest(dummy_video_file):
    files = {'file': ('test.mp4', open(dummy_video_file, 'rb'), 'video/mp4')}
    data = {'product_id': TEST_PRODUCT}
    
    response = requests.post(f"{API_BASE}/sessions/upload", files=files, data=data)
    assert response.status_code == 422
    assert "Missing quality_manifest" in response.text

def test_reject_demo_in_production(dummy_video_file):
    # This test assumes the server is NOT in LOCAL_DEV mode for validation
    # If the server is in LOCAL_DEV, it might pass. 
    # We should check the health check to see the env.
    health = requests.get(f"{API_BASE}/health").json()
    if health['env'] == 'local_dev':
        pytest.skip("Server is in local_dev, demo mode rejection not testable without config change.")

    manifest = {
        "is_demo": True,
        "product_profile": "generic",
        "coverage_summary": {"percent": 100, "maxGap": 0},
        "accepted_frame_count": 100,
        "total_frame_count": 100
    }
    
    files = {'file': ('test.mp4', open(dummy_video_file, 'rb'), 'video/mp4')}
    data = {
        'product_id': TEST_PRODUCT,
        'quality_manifest': json.dumps(manifest)
    }
    
    response = requests.post(f"{API_BASE}/sessions/upload", files=files, data=data)
    assert response.status_code == 422
    assert "Demo mode is only permitted in LOCAL_DEV" in response.text

def test_reject_bad_coverage(dummy_video_file):
    manifest = {
        "is_demo": False,
        "product_profile": "generic",
        "coverage_summary": {"percent": 50, "maxGap": 180},
        "accepted_frame_count": 100,
        "total_frame_count": 200,
        "rejection_stats": {}
    }
    
    files = {'file': ('test.mp4', open(dummy_video_file, 'rb'), 'video/mp4')}
    data = {
        'product_id': TEST_PRODUCT,
        'quality_manifest': json.dumps(manifest)
    }
    
    response = requests.post(f"{API_BASE}/sessions/upload", files=files, data=data)
    assert response.status_code == 422
    result = response.json()
    assert result['detail']['manifest_validation_status'] == "rejected"
    assert any("Coverage too low" in r for r in result['detail']['reasons'])

def test_accept_valid_manifest(dummy_video_file):
    manifest = {
        "is_demo": False,
        "product_profile": "generic",
        "coverage_summary": {"percent": 95, "maxGap": 20},
        "accepted_frame_count": 150,
        "total_frame_count": 200,
        "rejection_stats": {"Move slower (blur detected)": 10}
    }
    
    files = {'file': ('test.mp4', open(dummy_video_file, 'rb'), 'video/mp4')}
    data = {
        'product_id': TEST_PRODUCT,
        'quality_manifest': json.dumps(manifest)
    }
    
    response = requests.post(f"{API_BASE}/sessions/upload", files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result['manifest_validation_status'] == "passed"

if __name__ == "__main__":
    # For manual running
    pytest.main([__file__])
