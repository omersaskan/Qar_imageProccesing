import os
import shutil
import json
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from typing import Optional

from modules.operations.settings import Settings, AppEnvironment, settings
from modules.operations.api import app
from modules.capture_workflow.session_manager import SessionManager
from modules.shared_contracts.models import AssetStatus
from modules.operations.retention import RetentionService

# We use a TestClient for API checks
client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_settings():
    orig_env = settings.env
    orig_root = settings.data_root
    orig_key = settings.pilot_api_key
    yield
    settings.env = orig_env
    settings.data_root = orig_root
    settings.pilot_api_key = orig_key

@pytest.fixture
def pilot_mode():
    settings.env = AppEnvironment.PILOT
    settings.pilot_api_key = "supersecure"

@pytest.fixture
def dev_mode():
    settings.env = AppEnvironment.LOCAL_DEV

def test_api_security_pilot(pilot_mode):
    # 1. Health is public
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["env"] == "pilot"

    # 2. Ready is protected
    response = client.get("/api/ready")
    assert response.status_code == 401 

    # 3. Ready with correct key (X-API-KEY header)
    response = client.get("/api/ready", headers={"X-API-KEY": "supersecure"})
    assert response.status_code == 200

def test_api_security_dev(dev_mode):
    # Ready is public/optional in dev
    response = client.get("/api/ready")
    assert response.status_code == 200

def test_session_history_logic(tmp_path):
    sm = SessionManager(data_root=str(tmp_path))
    session = sm.create_session("h_001", "p_001", "op_001")
    
    # Transition 1
    sm.update_session_status("h_001", AssetStatus.CAPTURED)
    # Transition 2
    sm.update_session("h_001", new_status=AssetStatus.RECONSTRUCTED, note="Recon done")
    
    final_session = sm.get_session("h_001")
    assert len(final_session.history) == 3 
    assert final_session.history[1].to_status == "captured"
    assert final_session.history[2].to_status == "reconstructed"

def test_retention_pruning_policy(tmp_path):
    # Setup directories
    data_root = tmp_path
    sessions_dir = data_root / "sessions"
    captures_dir = data_root / "captures"
    reconstructions_dir = data_root / "reconstructions"
    sessions_dir.mkdir()
    captures_dir.mkdir()
    reconstructions_dir.mkdir()

    sm = SessionManager(data_root=str(data_root))
    
    # 1. Create a "PUBLISHED" session older than 3 days
    s1_id = "expired_pub"
    sm.create_session(s1_id, "p1", "op1")
    sess1 = sm.get_session(s1_id)
    sess1.status = AssetStatus.PUBLISHED
    sm.save_session(sess1)
    
    s1_file = sessions_dir / f"{s1_id}.json"
    old_time = (datetime.now() - timedelta(days=5)).timestamp()
    os.utime(s1_file, (old_time, old_time))
    
    c1_dir = captures_dir / s1_id
    (c1_dir / "reports").mkdir(parents=True, exist_ok=True)
    (c1_dir / "frames").mkdir(parents=True, exist_ok=True)
    (c1_dir / "reports" / "manifest.json").touch()
    (c1_dir / "frames" / "f1.jpg").touch()

    # 2. Run retention
    settings.data_root = str(data_root)
    rs = RetentionService(data_root=str(data_root))
    # We want to trace the decision
    rs.run_cleanup()

    # Verifications
    assert not (c1_dir / "frames" / "f1.jpg").exists()
    assert (c1_dir / "reports" / "manifest.json").exists()

def test_structured_logging_output():
    import logging
    from modules.operations.logging_config import JsonFormatter
    
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="Test Message", args=(), exc_info=None
    )
    setattr(record, "stage", "test_stage")
    setattr(record, "duration_ms", 123)
    
    json_out = json.loads(formatter.format(record))
    assert json_out["message"] == "Test Message"
    assert json_out["stage"] == "test_stage"
    assert json_out["duration_ms"] == 123
