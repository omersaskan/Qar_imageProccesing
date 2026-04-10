import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from modules.operations.api import app
from modules.operations.settings import settings, AppEnvironment

client = TestClient(app)
AUTH_HEADERS = {"X-API-KEY": "supersecure"}

@pytest.fixture(autouse=True)
def reset_settings():
    orig_env = settings.env
    orig_strict = settings.strict_ml_segmentation
    orig_key = settings.pilot_api_key
    
    settings.pilot_api_key = "supersecure"
    yield
    settings.env = orig_env
    settings.strict_ml_segmentation = orig_strict
    settings.pilot_api_key = orig_key

def test_ready_reports_missing_deps_pilot():
    # Simulate Pilot mode with missing rembg
    settings.env = AppEnvironment.PILOT
    
    with patch("importlib.util.find_spec") as mock_find:
        # Mock rembg as missing, onnxruntime as missing, but fast_simplification as present
        def side_effect(name):
            if name in ["rembg", "onnxruntime"]: return None
            return MagicMock()
        mock_find.side_effect = side_effect
        
        response = client.get("/api/ready", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_ready"
        # Search for the string in any of the issues
        assert any("Missing ML Segmentation dependencies" in issue for issue in data["issues"])
        assert data["dependencies"]["ml_segmentation_ready"] is False
        assert data["dependencies"]["critical_processing_ready"] is True

def test_upload_blocked_missing_deps_pilot():
    settings.env = AppEnvironment.PILOT
    
    with patch("importlib.util.find_spec") as mock_find:
        mock_find.return_value = None # All missing
        
        # We need to multi-part form for upload
        files = {"file": ("test.mp4", b"dummy_content", "video/mp4")}
        data = {"product_id": "p1", "operator_id": "op1"}
        
        response = client.post("/api/sessions/upload", data=data, files=files, headers=AUTH_HEADERS)
        assert response.status_code == 503
        assert "System Environment Incomplete" in response.json()["detail"]

def test_ready_allows_missing_deps_dev():
    settings.env = AppEnvironment.LOCAL_DEV
    
    with patch("importlib.util.find_spec") as mock_find:
        mock_find.return_value = None # Missing but allowed in dev
        
        # In DEV, key is optional usually, but we include it anyway or check bypass
        response = client.get("/api/ready") 
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready" # In dev we allow it
        assert any("Missing ML Segmentation dependencies" in issue for issue in data["issues"])
