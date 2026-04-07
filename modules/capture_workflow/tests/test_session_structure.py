import pytest
import os
from pathlib import Path
from modules.capture_workflow.session_manager import SessionManager

def test_session_manager_creates_all_folders(tmp_path):
    data_dir = tmp_path / "data"
    manager = SessionManager(data_root=str(data_dir))
    
    # 1. Create session
    session_id = "test_folders_sess"
    manager.create_session(session_id, "prod_1", "op_1")
    
    # 2. Check folders
    capture_dir = data_dir / "captures" / session_id
    assert (capture_dir / "video").is_dir()
    assert (capture_dir / "frames").is_dir()
    assert (capture_dir / "reports").is_dir()
    
    # 3. Check session file
    assert (data_dir / "sessions" / f"{session_id}.json").exists()
