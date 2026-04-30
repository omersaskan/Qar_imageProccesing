import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2
import io
import json
from modules.operations.api import app
from modules.operations.settings import settings

client = TestClient(app)

def get_test_image():
    # Create a simple white square image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode(".jpg", img)
    return io.BytesIO(img_encoded.tobytes())

@pytest.fixture(autouse=True)
def reset_settings():
    # Store original settings
    orig_enabled = settings.sam_mask_preview_enabled
    orig_provider = settings.segmentation_preview_provider
    orig_confidence = settings.sam_mask_min_confidence
    
    yield
    
    # Restore original settings
    settings.sam_mask_preview_enabled = orig_enabled
    settings.segmentation_preview_provider = orig_provider
    settings.sam_mask_min_confidence = orig_confidence

def test_mask_preview_disabled():
    settings.sam_mask_preview_enabled = False
    
    response = client.post(
        "/api/ar/mask-preview",
        files={"file": ("test.jpg", get_test_image(), "image/jpeg")},
        headers={"X-API-Key": settings.pilot_api_key} if not settings.is_dev else {}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["fallback_used"] is True
    assert data["mask"] == []

def test_mask_preview_legacy_provider():
    settings.sam_mask_preview_enabled = True
    settings.segmentation_preview_provider = "legacy"
    
    response = client.post(
        "/api/ar/mask-preview",
        files={"file": ("test.jpg", get_test_image(), "image/jpeg")},
        headers={"X-API-Key": settings.pilot_api_key} if not settings.is_dev else {}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "legacy"
    assert data["mask_format"] == "polygon"
    assert "mask" in data
    assert isinstance(data["mask"], list)
    assert data["confidence"] < 0.75 # Legacy is low confidence by default

def test_mask_preview_invalid_image():
    settings.sam_mask_preview_enabled = True
    
    response = client.post(
        "/api/ar/mask-preview",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        headers={"X-API-Key": settings.pilot_api_key} if not settings.is_dev else {}
    )
    
    # Strictly require 400 for invalid image data
    assert response.status_code == 400

def test_mask_preview_scaffold_providers():
    settings.sam_mask_preview_enabled = True
    
    for p in ["sam2", "sam3"]:
        settings.segmentation_preview_provider = p
        response = client.post(
            "/api/ar/mask-preview",
            files={"file": ("test.jpg", get_test_image(), "image/jpeg")},
            headers={"X-API-Key": settings.pilot_api_key} if not settings.is_dev else {}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == p
        assert data["fallback_used"] is True # Scaffolds should return fallback=True
