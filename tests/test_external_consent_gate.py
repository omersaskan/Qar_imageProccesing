import pytest
from fastapi.testclient import TestClient
from modules.operations.api import app
from modules.operations.settings import settings

client = TestClient(app)

def test_external_provider_consent_required():
    """Verify that external providers require explicit consent."""
    # 1. Enable Rodin for testing
    settings.rodin_enabled = True
    settings.rodin_api_key = "test_key"
    
    # 2. Upload an image to get a session
    with open("tests/test_data/test_image.jpg", "rb") as f:
        resp = client.post(
            "/api/ai-3d/upload",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"provider": "rodin"}
        )
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    
    # 3. Attempt process WITHOUT consent
    resp = client.post(
        f"/api/ai-3d/process/{session_id}",
        json={"options": {"external_provider_consent": False}}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "external_provider_consent_required"

def test_sf3d_does_not_require_consent():
    """Verify that the default SF3D provider does NOT require external consent."""
    # 1. Upload an image for SF3D
    with open("tests/test_data/test_image.jpg", "rb") as f:
        resp = client.post(
            "/api/ai-3d/upload",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"provider": "sf3d"}
        )
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    
    # 2. Process WITHOUT consent (should still work or fail for other reasons, not 400)
    # We use a mock to avoid running real SF3D if possible, but here we just check status != 400
    resp = client.post(
        f"/api/ai-3d/process/{session_id}",
        json={"options": {"external_provider_consent": False}}
    )
    # It might be 500 if SF3D isn't set up, but it shouldn't be 400 consent error
    assert resp.status_code != 400
