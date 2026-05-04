import pytest
from pathlib import Path
from modules.ai_3d_generation.rodin_provider import RodinProvider
from modules.operations.settings import settings

def test_rodin_provider_lifecycle_mocked(tmp_path, monkeypatch):
    """Verify the mocked lifecycle of RodinProvider."""
    # 1. Setup
    monkeypatch.setattr(settings, "env", "local_dev")
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "mock_key")
    monkeypatch.setattr(settings, "rodin_mock_mode", True)
    
    provider = RodinProvider()
    
    # 2. Mock some inputs
    input_img = tmp_path / "input.jpg"
    from PIL import Image
    Image.new("RGB", (1, 1)).save(input_img, "JPEG")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # 3. Generate (this will use the mocked methods)
    result = provider.generate(str(input_img), str(output_dir), {"external_provider_consent": True})
    
    # 4. Verify
    assert result["status"] == "ok"
    assert result["provider"] == "rodin"
    assert "mock_rodin_task_123" in result["output_path"]
    assert result["metadata"]["external_task_id"] == "mock_rodin_task_123"
    assert result["metadata"]["external_status"] == "succeeded"
    assert result["metadata"]["external_provider"] is True

def test_rodin_disabled_returns_unavailable(monkeypatch):
    """Verify that Rodin returns unavailable when disabled."""
    monkeypatch.setattr(settings, "rodin_enabled", False)
    provider = RodinProvider()
    
    avail, reason = provider.is_available()
    assert avail is False
    assert "disabled" in reason.lower()

def test_rodin_missing_key_returns_unavailable(monkeypatch):
    """Verify that Rodin returns unavailable when API key is missing."""
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "")
    provider = RodinProvider()
    
    avail, reason = provider.is_available()
    assert avail is False
    assert "API key is missing" in reason

def test_rodin_mock_mode_false_returns_not_implemented(monkeypatch):
    """Verify that Rodin returns not_implemented when mock mode is false and no real API."""
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "test_key")
    monkeypatch.setattr(settings, "rodin_mock_mode", False)
    
    provider = RodinProvider()
    avail, reason = provider.is_available()
    assert avail is False
    assert reason == "rodin_real_api_not_implemented"

def test_rodin_mock_mode_prohibited_in_production(monkeypatch):
    """Verify that Rodin mock mode is prohibited in non-local_dev environments."""
    monkeypatch.setattr(settings, "env", "production")
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "test_key")
    monkeypatch.setattr(settings, "rodin_mock_mode", True)
    
    provider = RodinProvider()
    avail, reason = provider.is_available()
    assert avail is False
    assert "prohibited" in reason.lower()
