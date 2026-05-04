import pytest
import json
import logging
from pathlib import Path
from fastapi.testclient import TestClient
from modules.operations.api import app
from modules.operations.settings import settings
from modules.ai_3d_generation.manifest import build_manifest

client = TestClient(app)

def test_api_keys_not_in_manifest():
    """Verify that external provider fields in manifest do not include secrets."""
    manifest = build_manifest(
        session_id="test_session",
        source_input_path="input.jpg",
        input_type="image",
        provider="rodin",
        provider_status="ok",
        model_name="gen-2",
        license_note="test license",
        selected_frame_path=None,
        prepared_image_path=None,
        preprocessing={},
        postprocessing={},
        quality_gate={},
        output_glb_path="output.glb",
        output_format="glb",
        preview_image_path=None,
        status="ok",
        warnings=[],
        errors=[],
        external_provider=True,
        external_task_id="task_123",
        external_provider_consent=True
    )
    
    # Stringify manifest
    m_str = json.dumps(manifest)
    
    # Verify no sensitive keywords are in the manifest structure itself
    # (Checking for the existence of the fields is fine, but they shouldn't contain the key)
    assert "rodin_api_key" not in m_str
    assert "Authorization" not in m_str
    assert "Bearer" not in m_str

def test_error_sanitization_leaks(caplog):
    """Verify that Authorization headers are sanitized from exception logs/responses."""
    settings.rodin_enabled = True
    settings.rodin_api_key = "SECRET_TOKEN_123"
    
    # We mock a failure that might include the secret in the message
    # e.g. "Failed to call API with Authorization: Bearer SECRET_TOKEN_123"
    
    # 1. Upload
    with open("tests/test_data/test_image.jpg", "rb") as f:
        resp = client.post(
            "/api/ai-3d/upload",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"provider": "rodin"}
        )
    session_id = resp.json()["session_id"]
    
    # 2. Mock the provider to raise a "leaky" exception
    from unittest.mock import patch
    with patch("modules.ai_3d_generation.pipeline.generate_ai_3d") as mock_gen:
        mock_gen.side_effect = Exception("Auth failed for Bearer SECRET_TOKEN_123")
        
        resp = client.post(
            f"/api/ai-3d/process/{session_id}",
            json={"options": {"external_provider_consent": True}}
        )
    
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "SECRET_TOKEN_123" not in detail
    assert "sensitive details redacted" in detail
    
    # Check logs too
    assert "SECRET_TOKEN_123" not in caplog.text
