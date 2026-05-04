import pytest
from pathlib import Path
from modules.ai_3d_generation.rodin_provider import RodinProvider
from modules.operations.settings import settings

def test_rodin_provider_lifecycle_mocked(tmp_path):
    """Verify the mocked lifecycle of RodinProvider."""
    # 1. Setup
    settings.rodin_enabled = True
    settings.rodin_api_key = "mock_key"
    provider = RodinProvider()
    
    # 2. Mock some inputs
    input_img = tmp_path / "input.jpg"
    input_img.write_text("fake_data")
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

def test_rodin_disabled_returns_unavailable():
    """Verify that Rodin returns unavailable when disabled."""
    settings.rodin_enabled = False
    provider = RodinProvider()
    
    avail, reason = provider.is_available()
    assert avail is False
    assert "disabled" in reason.lower()

def test_rodin_missing_key_returns_unavailable():
    """Verify that Rodin returns unavailable when API key is missing."""
    settings.rodin_enabled = True
    settings.rodin_api_key = ""
    provider = RodinProvider()
    
    avail, reason = provider.is_available()
    assert avail is False
    assert "API key is missing" in reason
