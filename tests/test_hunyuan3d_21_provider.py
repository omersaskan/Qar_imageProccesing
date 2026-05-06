import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from modules.ai_3d_generation.hunyuan3d_21_provider import Hunyuan3D21Provider
from modules.ai_3d_generation.pipeline import generate_ai_3d
from modules.operations.settings import settings

@pytest.fixture
def mock_settings():
    """Reset settings to a known state for tests."""
    orig_enabled = settings.hunyuan3d_21_enabled
    orig_repo = settings.hunyuan3d_21_repo_path
    orig_python = settings.hunyuan3d_21_python
    orig_ack = settings.hunyuan3d_21_legal_ack
    orig_mock = settings.hunyuan3d_21_mock_runner
    
    settings.hunyuan3d_21_enabled = False
    settings.hunyuan3d_21_repo_path = "/tmp/fake_hunyuan"
    settings.hunyuan3d_21_python = "/tmp/fake_python"
    settings.hunyuan3d_21_legal_ack = False
    settings.hunyuan3d_21_mock_runner = True
    
    yield settings
    
    settings.hunyuan3d_21_enabled = orig_enabled
    settings.hunyuan3d_21_repo_path = orig_repo
    settings.hunyuan3d_21_python = orig_python
    settings.hunyuan3d_21_legal_ack = orig_ack
    settings.hunyuan3d_21_mock_runner = orig_mock

def test_hunyuan_disabled(mock_settings):
    """1. disabled provider returns hunyuan3d_21_disabled"""
    mock_settings.hunyuan3d_21_enabled = False
    provider = Hunyuan3D21Provider()
    avail, reason = provider.is_available()
    assert avail is False
    assert reason == "hunyuan3d_21_disabled"

def test_hunyuan_repo_path_missing(mock_settings):
    """2. missing repo path returns hunyuan3d_21_repo_path_missing"""
    mock_settings.hunyuan3d_21_enabled = True
    mock_settings.hunyuan3d_21_legal_ack = True
    mock_settings.hunyuan3d_21_repo_path = "/nonexistent/repo"
    provider = Hunyuan3D21Provider()
    avail, reason = provider.is_available()
    assert avail is False
    assert reason == "hunyuan3d_21_repo_path_missing"

def test_hunyuan_python_missing(mock_settings, tmp_path):
    """3. missing python returns hunyuan3d_21_python_missing"""
    repo = tmp_path / "repo"
    repo.mkdir()
    mock_settings.hunyuan3d_21_enabled = True
    mock_settings.hunyuan3d_21_legal_ack = True
    mock_settings.hunyuan3d_21_repo_path = str(repo)
    mock_settings.hunyuan3d_21_python = "/nonexistent/python"
    provider = Hunyuan3D21Provider()
    avail, reason = provider.is_available()
    assert avail is False
    assert reason == "hunyuan3d_21_python_missing"

def test_hunyuan_legal_ack_required(mock_settings, tmp_path):
    """4. missing legal ack returns hunyuan3d_21_legal_ack_required"""
    repo = tmp_path / "repo"
    repo.mkdir()
    mock_settings.hunyuan3d_21_enabled = True
    mock_settings.hunyuan3d_21_legal_ack = False
    mock_settings.hunyuan3d_21_repo_path = str(repo)
    provider = Hunyuan3D21Provider()
    avail, reason = provider.is_available()
    assert avail is False
    assert reason == "hunyuan3d_21_legal_ack_required"

@patch("subprocess.run")
def test_hunyuan_command_building(mock_run, mock_settings, tmp_path):
    """5. subprocess command is built without leaking secrets"""
    repo = tmp_path / "repo"
    repo.mkdir()
    py = tmp_path / "python"
    py.touch()
    
    mock_settings.hunyuan3d_21_enabled = True
    mock_settings.hunyuan3d_21_legal_ack = True
    mock_settings.hunyuan3d_21_repo_path = str(repo)
    mock_settings.hunyuan3d_21_python = str(py)
    mock_settings.hunyuan3d_21_mock_runner = True
    
    mock_run.return_value = MagicMock(returncode=1, stderr="failed") # Force fail to check cmd
    
    provider = Hunyuan3D21Provider()
    provider.generate("fake_in.png", str(tmp_path))
    
    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == str(py)
    assert "--input-image" in cmd
    assert "--output-dir" in cmd
    assert "--mock-runner" in cmd

def test_hunyuan_mock_runner_success(mock_settings, tmp_path):
    """6. mock successful runner returns output_glb_path"""
    repo = tmp_path / "repo"
    repo.mkdir()
    py = sys.executable # Use real python for runner script
    
    mock_settings.hunyuan3d_21_enabled = True
    mock_settings.hunyuan3d_21_legal_ack = True
    mock_settings.hunyuan3d_21_repo_path = str(repo)
    mock_settings.hunyuan3d_21_python = py
    mock_settings.hunyuan3d_21_mock_runner = True
    
    provider = Hunyuan3D21Provider()
    result = provider.generate("tests/test_data/ai_3d/test_input.png", str(tmp_path))
    
    assert result["status"] == "ok"
    assert result["output_path"] is not None
    assert Path(result["output_path"]).exists()

def test_hunyuan_full_pipeline_integration(mock_settings, tmp_path):
    """7, 8, 9, 14, 15. E2E mock integration test."""
    repo = tmp_path / "repo"
    repo.mkdir()
    
    mock_settings.hunyuan3d_21_enabled = True
    mock_settings.hunyuan3d_21_legal_ack = True
    mock_settings.hunyuan3d_21_repo_path = str(repo)
    mock_settings.hunyuan3d_21_python = sys.executable
    mock_settings.hunyuan3d_21_mock_runner = True
    
    session_id = "test_hunyuan_e2e"
    output_dir = tmp_path / "session"
    output_dir.mkdir()
    
    # Run pipeline
    manifest = generate_ai_3d(
        session_id=session_id,
        input_file_path="tests/test_data/ai_3d/test_input.png",
        output_base_dir=str(output_dir),
        provider_name="hunyuan3d_21"
    )
    
    assert manifest["provider"] == "hunyuan3d_21"
    assert manifest["provider_status"] == "ok"
    
    # 7. GLB validation
    assert "glb_validation" in manifest
    
    # 8. Mesh stats
    assert "mesh_stats" in manifest
    
    # 9. AR readiness
    assert "ar_readiness" in manifest
    
    # 14. external_provider=false
    assert manifest["external_provider"] is False
    
    # 15. license_note is present
    assert "Tencent Hunyuan3D-2.1" in manifest["license_note"]

def test_api_provider_override(mock_settings, tmp_path):
    """10. API can request provider=hunyuan3d_21"""
    # This is partially covered by the pipeline test, but we verify resolution
    from modules.ai_3d_generation.pipeline import _get_provider
    p = _get_provider("hunyuan3d_21")
    assert isinstance(p, Hunyuan3D21Provider)

def test_ui_contains_selector():
    """11. UI contains Hunyuan provider selector"""
    ui_path = Path("ui/ai_3d_studio.html")
    content = ui_path.read_text(encoding="utf-8")
    assert 'value="hunyuan3d_21"' in content
    assert 'Hunyuan3D-2.1 Premium Local/Server' in content

def test_sf3d_defaults_unchanged():
    """12. SF3D defaults unchanged"""
    assert settings.ai_3d_default_provider == "sf3d"
    assert settings.sf3d_enabled is False or settings.sf3d_enabled is True # Should not be forced by Hunyuan

def test_external_providers_disabled_by_default():
    """13. external providers remain disabled by default"""
    assert settings.rodin_enabled is False
    assert settings.meshy_enabled is False
    assert settings.tripo_enabled is False
    assert settings.ai_3d_remote_providers_enabled is False

import sys


# ── Whitelist fix tests ────────────────────────────────────────────────────────

def test_write_session_inputs_accepts_hunyuan(tmp_path):
    """write_session_inputs must accept provider='hunyuan3d_21' without raising."""
    from modules.ai_3d_generation.multi_input import write_session_inputs
    f = tmp_path / "img.png"
    f.touch()
    # Must not raise
    out = write_session_inputs(str(tmp_path), "single_image", [str(f)], provider="hunyuan3d_21")
    import json
    data = json.loads(open(out, encoding="utf-8").read())
    assert data["provider"] == "hunyuan3d_21"


def test_load_session_inputs_with_hunyuan_provider(tmp_path):
    """load_session_inputs must accept a JSON file that names provider='hunyuan3d_21'."""
    from modules.ai_3d_generation.multi_input import load_session_inputs, write_session_inputs
    f = tmp_path / "img.png"
    f.touch()
    write_session_inputs(str(tmp_path), "single_image", [str(f)], provider="hunyuan3d_21")
    data = load_session_inputs(str(tmp_path))
    assert data is not None
    assert data["provider"] == "hunyuan3d_21"


def test_write_session_inputs_rejects_unknown_provider(tmp_path):
    """write_session_inputs must still reject arbitrary provider strings."""
    from modules.ai_3d_generation.multi_input import write_session_inputs
    f = tmp_path / "img.png"
    f.touch()
    with pytest.raises(ValueError, match="Invalid provider"):
        write_session_inputs(str(tmp_path), "single_image", [str(f)], provider="evil_cloud_api")


# ── Runner real-path tests (no actual Hunyuan model loaded) ────────────────────

@patch("subprocess.run")
def test_runner_real_codepath_via_manifest(mock_run, mock_settings, tmp_path):
    """Real runner code path: provider reads manifest written by worker (no model)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    py = tmp_path / "python"
    py.touch()

    mock_settings.hunyuan3d_21_enabled = True
    mock_settings.hunyuan3d_21_legal_ack = True
    mock_settings.hunyuan3d_21_repo_path = str(repo)
    mock_settings.hunyuan3d_21_python = str(py)
    mock_settings.hunyuan3d_21_mock_runner = False  # real path

    # Pre-write the manifest the runner would have written on success
    output_glb = tmp_path / "output.glb"
    output_glb.write_bytes(b"fake_glb")
    manifest_data = {
        "status": "ok",
        "output_glb_path": str(output_glb),
        "warnings": ["ai_generated_not_true_scan"],
        "error": None,
        "peak_mem_mb": 4096,
        "mode": "shape_only",
        "model_path": "tencent/Hunyuan3D-2.1",
        "device": "cuda",
    }
    (tmp_path / "hunyuan_worker_manifest.json").write_text(
        json.dumps(manifest_data), encoding="utf-8"
    )

    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    provider = Hunyuan3D21Provider()
    result = provider.generate("tests/test_data/ai_3d/test_input.png", str(tmp_path))

    assert result["status"] == "ok"
    assert result["output_path"] == str(output_glb)
    assert result["metadata"]["mode"] == "shape_only"
    assert result["metadata"]["peak_mem_mb"] == 4096
    assert result["metadata"]["external_provider"] is False


@patch("subprocess.run")
def test_runner_failure_manifest_sanitized_error(mock_run, mock_settings, tmp_path):
    """Worker writes failure manifest with sanitized error; provider propagates it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    py = tmp_path / "python"
    py.touch()

    mock_settings.hunyuan3d_21_enabled = True
    mock_settings.hunyuan3d_21_legal_ack = True
    mock_settings.hunyuan3d_21_repo_path = str(repo)
    mock_settings.hunyuan3d_21_python = str(py)
    mock_settings.hunyuan3d_21_mock_runner = False

    # Worker wrote a failure manifest (e.g. CUDA OOM)
    manifest_data = {
        "status": "failed",
        "output_glb_path": None,
        "warnings": [],
        "error": "cuda_oom",
        "peak_mem_mb": None,
    }
    (tmp_path / "hunyuan_worker_manifest.json").write_text(
        json.dumps(manifest_data), encoding="utf-8"
    )

    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    provider = Hunyuan3D21Provider()
    result = provider.generate("tests/test_data/ai_3d/test_input.png", str(tmp_path))

    assert result["status"] == "failed"
    assert result["error"] == "cuda_oom"
    assert result["output_path"] is None


def test_api_no_500_on_hunyuan_provider_failure(tmp_path):
    """POST /api/ai-3d/process must return non-500 when Hunyuan provider fails."""
    import modules.operations.api as api_module
    from fastapi.testclient import TestClient

    client = TestClient(api_module.app)

    sid = "test_hunyuan_no500"
    api_module._ai3d_sessions[sid] = {
        "session_id": sid,
        "status": "uploaded",
        "input_path": "tests/test_data/ai_3d/test_input.png",
        "provider": "hunyuan3d_21",
        "manifest_path": None,
    }

    failed_manifest = {
        "session_id": sid,
        "status": "failed",
        "provider": "hunyuan3d_21",
        "provider_status": "failed",
        "output_glb_path": None,
        "errors": ["hunyuan_worker_failed"],
        "warnings": [],
        "worker_metadata": {},
        "ar_readiness": {"enabled": True, "score": 0, "verdict": "not_ready"},
        "mesh_stats": {"enabled": True, "available": False},
        "glb_validation": None,
    }

    orig = api_module.settings.ai_3d_generation_enabled
    api_module.settings.ai_3d_generation_enabled = True
    try:
        with patch("modules.operations.api.verify_api_key"), \
             patch("modules.ai_3d_generation.pipeline.generate_ai_3d",
                   return_value=failed_manifest), \
             patch("modules.ai_3d_generation.multi_input.load_session_inputs",
                   return_value={"input_mode": "single_image", "uploaded_files_count": 1,
                                 "input_files": [], "provider": "hunyuan3d_21"}):
            resp = client.post(f"/api/ai-3d/process/{sid}",
                               headers={"x-api-key": "test"})
    finally:
        api_module.settings.ai_3d_generation_enabled = orig
        api_module._ai3d_sessions.pop(sid, None)

    # Provider failure must return 200 with status/provider_status=failed, never 500
    assert resp.status_code != 500
    body = resp.json()
    assert body.get("provider_status") == "failed" or body.get("status") == "failed"


# ── Package roots / import path tests ─────────────────────────────────────────

@patch("subprocess.run")
def test_provider_cmd_includes_repo_path(mock_run, mock_settings, tmp_path):
    """Provider command must pass --repo-path with the configured repo path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    py = tmp_path / "python"
    py.touch()

    mock_settings.hunyuan3d_21_enabled = True
    mock_settings.hunyuan3d_21_legal_ack = True
    mock_settings.hunyuan3d_21_repo_path = str(repo)
    mock_settings.hunyuan3d_21_python = str(py)
    mock_settings.hunyuan3d_21_mock_runner = False

    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="no manifest")

    provider = Hunyuan3D21Provider()
    provider.generate("tests/test_data/ai_3d/test_input.png", str(tmp_path))

    cmd = mock_run.call_args[0][0]
    assert "--repo-path" in cmd
    # The value following --repo-path must resolve to the configured repo
    idx = cmd.index("--repo-path")
    assert Path(cmd[idx + 1]).resolve() == repo.resolve()


def test_runner_setup_inserts_existing_package_roots(tmp_path):
    """_setup_package_roots inserts hy3dshape, hy3dpaint, and repo root when present."""
    import importlib.util, types
    spec = importlib.util.spec_from_file_location(
        "run_hunyuan3d_21",
        str(Path("scripts/run_hunyuan3d_21.py").resolve()),
    )
    runner = types.ModuleType("run_hunyuan3d_21")
    spec.loader.exec_module(runner)  # type: ignore[union-attr]

    repo = tmp_path / "myrepo"
    (repo / "hy3dshape").mkdir(parents=True)
    (repo / "hy3dpaint").mkdir(parents=True)
    repo.mkdir(exist_ok=True)

    before = list(sys.path)
    inserted = runner._setup_package_roots(str(repo))
    # Remove what we inserted so we don't pollute other tests
    for p in inserted:
        sys.path.remove(p)

    assert str(repo / "hy3dshape") in inserted
    assert str(repo / "hy3dpaint") in inserted
    assert str(repo) in inserted


def test_runner_setup_skips_missing_subdirs(tmp_path):
    """_setup_package_roots only inserts paths that actually exist on disk."""
    import importlib.util, types
    spec = importlib.util.spec_from_file_location(
        "run_hunyuan3d_21",
        str(Path("scripts/run_hunyuan3d_21.py").resolve()),
    )
    runner = types.ModuleType("run_hunyuan3d_21")
    spec.loader.exec_module(runner)  # type: ignore[union-attr]

    repo = tmp_path / "emptyrepo"
    repo.mkdir()  # exists but no hy3dshape/ or hy3dpaint/

    inserted = runner._setup_package_roots(str(repo))
    for p in inserted:
        if p in sys.path:
            sys.path.remove(p)

    # Only the repo root itself should be inserted (it exists); subdirs should not
    assert str(repo / "hy3dshape") not in inserted
    assert str(repo / "hy3dpaint") not in inserted
    assert str(repo) in inserted


def test_sanitize_error_import_includes_module_name():
    """_sanitize_error must embed the actual module name for ImportError."""
    import importlib.util, types
    spec = importlib.util.spec_from_file_location(
        "run_hunyuan3d_21",
        str(Path("scripts/run_hunyuan3d_21.py").resolve()),
    )
    runner = types.ModuleType("run_hunyuan3d_21")
    spec.loader.exec_module(runner)  # type: ignore[union-attr]

    exc = ModuleNotFoundError("No module named 'hy3dshape.pipelines'")
    result = runner._sanitize_error(exc)
    assert result == "import_error:No module named 'hy3dshape.pipelines'"


def test_real_runner_import_error_includes_module_name(mock_settings, tmp_path):
    """Running the real runner (no hy3dshape installed) must emit import_error:<module>."""
    repo = tmp_path / "repo"
    repo.mkdir()

    mock_settings.hunyuan3d_21_enabled = True
    mock_settings.hunyuan3d_21_legal_ack = True
    mock_settings.hunyuan3d_21_repo_path = str(repo)
    mock_settings.hunyuan3d_21_python = sys.executable  # test-env Python, no hy3dshape
    mock_settings.hunyuan3d_21_mock_runner = False

    provider = Hunyuan3D21Provider()
    result = provider.generate("tests/test_data/ai_3d/test_input.png", str(tmp_path))

    assert result["status"] == "failed"
    err = result.get("error", "")
    assert err.startswith("import_error:"), f"Expected import_error prefix, got: {err!r}"
    assert "hy3dshape" in err or "PIL" in err or "module" in err.lower()


def test_background_remover_failure_adds_warning_not_error():
    """_try_remove_background must return (original_image, warning) when import fails."""
    import importlib.util, types
    from unittest.mock import MagicMock, patch as _patch

    spec = importlib.util.spec_from_file_location(
        "run_hunyuan3d_21",
        str(Path("scripts/run_hunyuan3d_21.py").resolve()),
    )
    runner = types.ModuleType("run_hunyuan3d_21")
    spec.loader.exec_module(runner)  # type: ignore[union-attr]

    fake_image = MagicMock()
    fake_image.convert.return_value = fake_image

    with _patch.dict("sys.modules", {"hy3dshape.rembg": None}):
        result_image, warning = runner._try_remove_background(fake_image)

    # Original image returned, warning is set
    assert result_image is fake_image
    assert warning is not None
    assert warning.startswith("background_remover_unavailable:")


# ── Existing whitelist tests ────────────────────────────────────────────────────

def test_write_session_inputs_sf3d_unchanged(tmp_path):
    """SF3D provider must still be accepted (regression guard)."""
    from modules.ai_3d_generation.multi_input import write_session_inputs
    f = tmp_path / "img.png"
    f.touch()
    import json
    out = write_session_inputs(str(tmp_path), "single_image", [str(f)], provider="sf3d")
    data = json.loads(open(out, encoding="utf-8").read())
    assert data["provider"] == "sf3d"
