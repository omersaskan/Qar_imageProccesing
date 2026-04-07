import pytest
from pathlib import Path
from capture_workflow.session_manager import SessionManager
from shared_contracts.lifecycle import AssetStatus

def test_session_creation_persistence(tmp_path):
    # Use tmp_path for isolated testing
    manager = SessionManager(data_root=str(tmp_path))
    session_id = "S_TEST_001"
    
    session = manager.create_session(session_id, "P1", "O1")
    
    # Check if files exist
    assert (tmp_path / "sessions" / f"{session_id}.json").exists()
    assert (tmp_path / "captures" / session_id / "frames").is_dir()
    
    # Reload from storage
    loaded = manager.get_session(session_id)
    assert loaded.session_id == session_id
    assert loaded.status == AssetStatus.CREATED

def test_session_status_transition(tmp_path):
    manager = SessionManager(data_root=str(tmp_path))
    session_id = "S_TEST_002"
    manager.create_session(session_id, "P1", "O1")
    
    # Valid transition
    updated = manager.update_session_status(session_id, AssetStatus.CAPTURED)
    assert updated.status == AssetStatus.CAPTURED
    
    # Verify persistence
    loaded = manager.get_session(session_id)
    assert loaded.status == AssetStatus.CAPTURED
