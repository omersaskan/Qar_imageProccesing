import pytest
from fastapi.testclient import TestClient
from modules.operations.api import app
from modules.operations.settings import settings

client = TestClient(app)

@pytest.fixture
def clean_settings(monkeypatch):
    """Fixture to ensure settings are reset after each test."""
    # We don't need to do much here if we use monkeypatch.setattr in every test
    pass

def test_ai_3d_generation_disabled_returns_503(monkeypatch):
    """Verify that process returns 503 if AI_3D_GENERATION_ENABLED=false."""
    monkeypatch.setattr(settings, "ai_3d_generation_enabled", False)
    
    # 1. Upload still works (as per requirement)
    with open("tests/test_data/test_image.jpg", "rb") as f:
        resp = client.post(
            "/api/ai-3d/upload",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"provider": "sf3d"}
        )
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    
    # 2. Process returns 503
    resp = client.post(f"/api/ai-3d/process/{session_id}")
    assert resp.status_code == 503
    assert "disabled" in resp.json()["detail"].lower()

def test_external_provider_consent_required(monkeypatch):
    """Verify that external providers require explicit consent."""
    monkeypatch.setattr(settings, "ai_3d_generation_enabled", True)
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "test_key")
    monkeypatch.setattr(settings, "rodin_mock_mode", True)
    
    # 1. Upload an image to get a session
    with open("tests/test_data/test_image.jpg", "rb") as f:
        resp = client.post(
            "/api/ai-3d/upload",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"provider": "rodin"}
        )
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    
    # 2. Attempt process WITHOUT consent
    resp = client.post(
        f"/api/ai-3d/process/{session_id}",
        json={"options": {"external_provider_consent": False}}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "external_provider_consent_required"

    # 3. Attempt process WITH consent (should proceed to pipeline)
    resp = client.post(
        f"/api/ai-3d/process/{session_id}",
        json={"options": {"external_provider_consent": True}}
    )
    # Status might be 500 if other things fail, but NOT 400 consent error
    assert resp.status_code != 400

def test_sf3d_does_not_require_consent(monkeypatch):
    """Verify that the default SF3D provider does NOT require external consent."""
    monkeypatch.setattr(settings, "ai_3d_generation_enabled", True)
    monkeypatch.setattr(settings, "sf3d_enabled", True)
    
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
    resp = client.post(
        f"/api/ai-3d/process/{session_id}",
        json={"options": {"external_provider_consent": False}}
    )
    assert resp.status_code != 400

def test_provider_persistence_and_mismatch(monkeypatch):
    """Verify that provider is persisted and mismatch is caught."""
    monkeypatch.setattr(settings, "ai_3d_generation_enabled", True)
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "test_key")
    
    # 1. Upload with rodin
    with open("tests/test_data/test_image.jpg", "rb") as f:
        resp = client.post(
            "/api/ai-3d/upload",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"provider": "rodin"}
        )
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    
    # 2. Process with sf3d (mismatch)
    resp = client.post(
        f"/api/ai-3d/process/{session_id}",
        json={"options": {"provider": "sf3d"}}
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "provider_mismatch"

    # 3. Process with rodin (matches)
    resp = client.post(
        f"/api/ai-3d/process/{session_id}",
        json={"options": {"provider": "rodin", "external_provider_consent": True}}
    )
    assert resp.status_code != 400
