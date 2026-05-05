"""
Phase 4C tests — mesh_stats and ar_readiness in API process response.

Verifies that POST /api/ai-3d/process/{session_id} includes both
`mesh_stats` and `ar_readiness` as explicit top-level keys, in addition
to them being nested inside `manifest`.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from modules.operations.api import app
import modules.operations.api as api_module

client = TestClient(app)


@pytest.fixture(autouse=True)
def _bypass_auth():
    with patch("modules.operations.api.verify_api_key"):
        yield


@pytest.fixture(autouse=True)
def _enable_ai3d():
    original = api_module.settings.ai_3d_generation_enabled
    api_module.settings.ai_3d_generation_enabled = True
    yield
    api_module.settings.ai_3d_generation_enabled = original


def _inject_session(session_id: str, input_path: str):
    """Register a fake uploaded session in the API's in-memory store."""
    api_module._ai3d_sessions[session_id] = {
        "status": "uploaded",
        "input_path": input_path,
        "provider": "sf3d",
    }


def _mock_manifest(glb_path: str) -> dict:
    return {
        "session_id": "test_sess",
        "status": "review",
        "provider": "sf3d",
        "provider_status": "ok",
        "execution_mode": "wsl_subprocess",
        "output_glb_path": glb_path,
        "output_size_bytes": 1024,
        "peak_mem_mb": 500.0,
        "worker_metadata": {"device": "cuda", "texture_resolution": 1024},
        "input_type": "single_image",
        "input_mode": "single_image",
        "candidate_count": 1,
        "selected_candidate_id": "cand_001",
        "candidate_ranking": [],
        "warnings": [],
        "errors": [],
        "review_required": True,
        "is_true_scan": False,
        "geometry_confidence": "estimated",
        "model_name": "stable-fast-3d",
        "quality_mode": "high",
        "resolved_quality": {"input_size": 1024},
        "preprocessing": {},
        "postprocessing": {},
        "quality_gate": {"verdict": "review"},
        "missing_outputs": [],
        "mesh_stats": {
            "enabled": True,
            "available": True,
            "vertex_count": 15000,
            "face_count": 28000,
            "geometry_count": 1,
            "error": None,
        },
        "ar_readiness": {
            "enabled": True,
            "score": 90,
            "verdict": "mobile_ready",
            "checks": {},
            "warnings": [],
            "recommendations": [],
        },
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_process_response_includes_mesh_stats_top_level(tmp_path):
    sid = "test_4c_mesh"
    inp = str(tmp_path / "upload.png")
    Path(inp).touch()
    _inject_session(sid, inp)

    manifest = _mock_manifest(str(tmp_path / "output.glb"))

    with patch("modules.ai_3d_generation.pipeline.generate_ai_3d", return_value=manifest), \
         patch("modules.ai_3d_generation.multi_input.load_session_inputs", return_value=None):
        r = client.post(f"/api/ai-3d/process/{sid}", json={"options": {}})

    assert r.status_code == 200
    data = r.json()
    assert "mesh_stats" in data, "mesh_stats must be a top-level key in the process response"


def test_process_response_includes_ar_readiness_top_level(tmp_path):
    sid = "test_4c_ar"
    inp = str(tmp_path / "upload.png")
    Path(inp).touch()
    _inject_session(sid, inp)

    manifest = _mock_manifest(str(tmp_path / "output.glb"))

    with patch("modules.ai_3d_generation.pipeline.generate_ai_3d", return_value=manifest), \
         patch("modules.ai_3d_generation.multi_input.load_session_inputs", return_value=None):
        r = client.post(f"/api/ai-3d/process/{sid}", json={"options": {}})

    assert r.status_code == 200
    data = r.json()
    assert "ar_readiness" in data, "ar_readiness must be a top-level key in the process response"


def test_process_response_mesh_stats_values_match_manifest(tmp_path):
    sid = "test_4c_ms_vals"
    inp = str(tmp_path / "upload.png")
    Path(inp).touch()
    _inject_session(sid, inp)

    manifest = _mock_manifest(str(tmp_path / "output.glb"))

    with patch("modules.ai_3d_generation.pipeline.generate_ai_3d", return_value=manifest), \
         patch("modules.ai_3d_generation.multi_input.load_session_inputs", return_value=None):
        r = client.post(f"/api/ai-3d/process/{sid}", json={"options": {}})

    data = r.json()
    ms = data["mesh_stats"]
    assert ms["available"] is True
    assert ms["vertex_count"] == 15000
    assert ms["face_count"] == 28000
    assert ms["geometry_count"] == 1


def test_process_response_ar_readiness_values_match_manifest(tmp_path):
    sid = "test_4c_ar_vals"
    inp = str(tmp_path / "upload.png")
    Path(inp).touch()
    _inject_session(sid, inp)

    manifest = _mock_manifest(str(tmp_path / "output.glb"))

    with patch("modules.ai_3d_generation.pipeline.generate_ai_3d", return_value=manifest), \
         patch("modules.ai_3d_generation.multi_input.load_session_inputs", return_value=None):
        r = client.post(f"/api/ai-3d/process/{sid}", json={"options": {}})

    data = r.json()
    ar = data["ar_readiness"]
    assert ar["score"] == 90
    assert ar["verdict"] == "mobile_ready"
    assert ar["enabled"] is True


def test_process_response_mesh_stats_also_in_manifest(tmp_path):
    """mesh_stats appears both at top level AND nested inside manifest."""
    sid = "test_4c_nested"
    inp = str(tmp_path / "upload.png")
    Path(inp).touch()
    _inject_session(sid, inp)

    manifest = _mock_manifest(str(tmp_path / "output.glb"))

    with patch("modules.ai_3d_generation.pipeline.generate_ai_3d", return_value=manifest), \
         patch("modules.ai_3d_generation.multi_input.load_session_inputs", return_value=None):
        r = client.post(f"/api/ai-3d/process/{sid}", json={"options": {}})

    data = r.json()
    assert data["mesh_stats"] == data["manifest"]["mesh_stats"]
    assert data["ar_readiness"] == data["manifest"]["ar_readiness"]


def test_process_response_mesh_stats_none_when_absent(tmp_path):
    """When manifest has no mesh_stats, top-level key is present but null."""
    sid = "test_4c_absent"
    inp = str(tmp_path / "upload.png")
    Path(inp).touch()
    _inject_session(sid, inp)

    manifest = _mock_manifest(str(tmp_path / "output.glb"))
    del manifest["mesh_stats"]
    del manifest["ar_readiness"]

    with patch("modules.ai_3d_generation.pipeline.generate_ai_3d", return_value=manifest), \
         patch("modules.ai_3d_generation.multi_input.load_session_inputs", return_value=None):
        r = client.post(f"/api/ai-3d/process/{sid}", json={"options": {}})

    data = r.json()
    assert "mesh_stats" in data
    assert data["mesh_stats"] is None
    assert "ar_readiness" in data
    assert data["ar_readiness"] is None
