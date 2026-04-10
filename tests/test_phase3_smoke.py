import os
import pytest
from pathlib import Path
from modules.operations.settings import Settings, AppEnvironment
from modules.shared_contracts.models import CaptureSession, AssetStatus
from modules.capture_workflow.session_manager import SessionManager

def test_settings_profiles(tmp_path):
    # Test local_dev defaults
    s = Settings(_env_file=None, ENV="local_dev")
    assert s.env == AppEnvironment.LOCAL_DEV
    assert s.is_dev is True
    
    # Test pilot requirements
    dummy_bin = tmp_path / "colmap.exe"
    dummy_bin.touch()
    
    s = Settings(_env_file=None, ENV="pilot", PILOT_API_KEY="test_key", RECON_ENGINE_PATH=str(dummy_bin))
    assert s.env == AppEnvironment.PILOT
    s.validate_setup() # Should not raise
    
    # Test pilot failure without key
    with pytest.raises(ValueError, match="PILOT_API_KEY is mandatory"):
        Settings(_env_file=None, ENV="pilot", PILOT_API_KEY="", RECON_ENGINE_PATH=str(dummy_bin)).validate_setup()

def test_audit_trail(tmp_path):
    # Setup session manager with temp path
    sm = SessionManager(data_root=str(tmp_path))
    
    # 1. Create session
    session = sm.create_session("sess_001", "prod_001", "op_001")
    assert len(session.history) == 1
    assert session.history[0].from_status == "none"
    assert session.history[0].to_status == AssetStatus.CREATED.value
    
    # 2. Update status
    updated = sm.update_session_status("sess_001", AssetStatus.CAPTURED)
    assert len(updated.history) == 2
    assert updated.history[1].from_status == AssetStatus.CREATED.value
    assert updated.history[1].to_status == AssetStatus.CAPTURED.value
    
    # 3. Update with note
    updated_again = sm.update_session("sess_001", new_status=AssetStatus.FAILED, note="Test failure")
    assert len(updated_again.history) == 3
    assert updated_again.history[2].to_status == AssetStatus.FAILED.value
    assert updated_again.history[2].note == "Test failure"

if __name__ == "__main__":
    # Manual run if needed
    pytest.main([__file__])
