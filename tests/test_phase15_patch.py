import pytest
import json
from pathlib import Path
from modules.operations.settings import settings
from modules.ai_3d_generation.manifest import build_manifest
from modules.ai_3d_generation.sanitization import sanitize_text
from modules.ai_3d_generation.multi_input import load_session_inputs, write_session_inputs
from modules.ai_3d_generation.provider_base import _normalise_status

def test_sanitization_direct_replacement(monkeypatch):
    """Verify that actual configured secrets are redacted regardless of prefix."""
    monkeypatch.setattr(settings, "rodin_api_key", "SECRET123")
    monkeypatch.setattr(settings, "meshy_api_key", "KEY-DOT.SLASH/456")
    monkeypatch.setattr(settings, "pilot_api_key", "short")
    
    text = "Here is SECRET123 and another KEY-DOT.SLASH/456 and a short one."
    sanitized = sanitize_text(text)
    
    assert "SECRET123" not in sanitized
    assert "KEY-DOT.SLASH/456" not in sanitized
    assert "short" not in sanitized
    assert sanitized.count("[REDACTED]") == 3

def test_manifest_boundary_sanitization(monkeypatch):
    """Verify that build_manifest sanitizes error fields."""
    monkeypatch.setattr(settings, "rodin_api_key", "MY_SECRET_KEY")
    
    # Test sanitized_error
    m = build_manifest(
        session_id="s1", source_input_path="i.jpg", input_type="image",
        provider="rodin", provider_status="failed", model_name=None,
        license_note="", selected_frame_path=None, prepared_image_path=None,
        preprocessing={}, postprocessing={}, quality_gate={},
        output_glb_path=None, output_format="glb", preview_image_path=None,
        status="failed", warnings=[], errors=[],
        sanitized_error="Error with MY_SECRET_KEY"
    )
    assert "MY_SECRET_KEY" not in m["sanitized_error"]
    assert "[REDACTED]" in m["sanitized_error"]
    
    # Test provider_failure_reason
    m2 = build_manifest(
        session_id="s1", source_input_path="i.jpg", input_type="image",
        provider="rodin", provider_status="failed", model_name=None,
        license_note="", selected_frame_path=None, prepared_image_path=None,
        preprocessing={}, postprocessing={}, quality_gate={},
        output_glb_path=None, output_format="glb", preview_image_path=None,
        status="failed", warnings=[], errors=[],
        provider_failure_reason="Failed due to api_key=MY_SECRET_KEY"
    )
    assert "MY_SECRET_KEY" not in m2["provider_failure_reason"]
    assert "[REDACTED]" in m2["provider_failure_reason"]

def test_normalise_status_sanitization(monkeypatch):
    """Verify that _normalise_status sanitizes the original error."""
    monkeypatch.setattr(settings, "rodin_api_key", "SECRET_KEY_999")
    
    res = {
        "status": "unknown_status",
        "error": "Failed with Bearer SECRET_KEY_999"
    }
    normalized = _normalise_status(res, "rodin", "glb")
    assert normalized["status"] == "failed"
    assert "SECRET_KEY_999" not in normalized["error"]
    assert "[REDACTED]" in normalized["error"]

def test_multi_input_invalid_provider_strict(tmp_path):
    """Verify that load_session_inputs raises ValueError for invalid provider."""
    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    input_dir.mkdir(parents=True)
    
    session_inputs = {
        "input_mode": "single_image",
        "input_files": ["upload.jpg"],
        "provider": "invalid_provider_name"
    }
    (input_dir / "session_inputs.json").write_text(json.dumps(session_inputs))
    
    with pytest.raises(ValueError, match="Invalid provider"):
        load_session_inputs(str(session_dir))

def test_input_mode_normalization_in_manifest():
    """Verify that 'image' input_type is normalized to 'single_image' in manifest."""
    from modules.ai_3d_generation.pipeline import _build_failed_manifest
    from unittest.mock import MagicMock
    
    provider = MagicMock()
    provider.name = "sf3d"
    provider.license_note = "note"
    provider.output_format = "glb"
    
    # Mocking build_manifest to see what it receives
    with MagicMock() as mock_build:
        import modules.ai_3d_generation.pipeline as pipeline
        pipeline.build_manifest = mock_build
        
        _build_failed_manifest("s1", "i.jpg", "image", provider, [], [], None)
        
        # Check first call args
        args, kwargs = mock_build.call_args
        assert kwargs["input_type"] == "single_image"

def test_rodin_env_guard_robustness(monkeypatch):
    """Verify Rodin env guard handles enum or string."""
    from modules.ai_3d_generation.rodin_provider import RodinProvider
    from modules.operations.settings import AppEnvironment
    
    provider = RodinProvider()
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "key")
    monkeypatch.setattr(settings, "rodin_mock_mode", True)
    
    # Test string "production"
    monkeypatch.setattr(settings, "env", "production")
    avail, reason = provider.is_available()
    assert avail is False
    assert "prohibited" in reason
    
    # Test Enum AppEnvironment.PRODUCTION
    monkeypatch.setattr(settings, "env", AppEnvironment.PRODUCTION)
    avail, reason = provider.is_available()
    assert avail is False
    assert "prohibited" in reason
    
    # Test string "local_dev"
    monkeypatch.setattr(settings, "env", "local_dev")
    monkeypatch.setattr(settings, "rodin_mock_mode", True)
    # is_available will continue to check real api implemented if mock is false,
    # but here it should pass the env check.
    # It might still return False, "rodin_real_api_not_implemented" if mock is false,
    # but here mock is True.
    avail, reason = provider.is_available()
    assert avail is True
