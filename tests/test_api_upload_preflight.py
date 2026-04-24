import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import os
from unittest.mock import patch, MagicMock
from modules.operations.api import app
from modules.operations.settings import settings

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_auth():
    with patch("modules.operations.api.verify_api_key"):
        yield

def test_upload_empty_file(tmp_path):
    empty_file = tmp_path / "empty.mp4"
    empty_file.touch()
    
    with open(empty_file, "rb") as f:
        response = client.post(
            "/api/sessions/upload",
            data={"product_id": "prod_1", "operator_id": "op_1"},
            files={"file": ("empty.mp4", f, "video/mp4")}
        )
        
    assert response.status_code == 400
    assert "too small or empty" in response.json()["detail"].lower()

@patch("cv2.VideoCapture")
def test_upload_unreadable_video(mock_vc, tmp_path):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_vc.return_value = mock_cap
    
    dummy_file = tmp_path / "dummy.mp4"
    dummy_file.write_bytes(b"x" * (1024 * 1024)) # 1MB dummy
    
    with open(dummy_file, "rb") as f:
        response = client.post(
            "/api/sessions/upload",
            data={"product_id": "prod_1", "operator_id": "op_1"},
            files={"file": ("dummy.mp4", f, "video/mp4")}
        )
        
    assert response.status_code == 400
    assert "unreadable" in response.json()["detail"].lower()
    mock_cap.release.assert_called_once()

@patch("cv2.VideoCapture")
def test_upload_short_video(mock_vc, tmp_path):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        5: 30.0, # cv2.CAP_PROP_FPS
        7: 60,   # cv2.CAP_PROP_FRAME_COUNT (2.0s)
        3: 1080, # width
        4: 1080  # height
    }[prop]
    mock_vc.return_value = mock_cap
    
    dummy_file = tmp_path / "dummy.mp4"
    dummy_file.write_bytes(b"x" * (1024 * 1024))
    
    with patch.object(settings, 'min_video_duration_sec', 8.0):
        with open(dummy_file, "rb") as f:
            response = client.post(
                "/api/sessions/upload",
                data={"product_id": "prod_1", "operator_id": "op_1"},
                files={"file": ("dummy.mp4", f, "video/mp4")}
            )
            
    assert response.status_code == 400
    assert "too short" in response.json()["detail"].lower()
