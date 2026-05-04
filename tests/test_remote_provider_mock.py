"""
Remote provider mock tests — stabilization patch.

Tests cover:
  A) settings defaults: remote providers disabled, consent required
  B) provider safety: global switch, rodin disabled, mock mode prohibition, missing key
  C) consent gate: Rodin blocked without consent, can proceed with consent+mock+switch
  D) sanitization: no secrets in errors or manifests
  E) regression: SF3D single-image / multi-candidate still work
"""
import pytest
from pathlib import Path
from modules.ai_3d_generation.rodin_provider import RodinProvider
from modules.operations.settings import settings


# ── A) Settings defaults ────────────────────────────────────────────────────

def test_remote_providers_disabled_by_default():
    """AI_3D_REMOTE_PROVIDERS_ENABLED must default to False."""
    fresh = type(settings)(_env_file=None)
    assert fresh.ai_3d_remote_providers_enabled is False


def test_require_external_consent_true_by_default():
    """AI_3D_REQUIRE_EXTERNAL_CONSENT must default to True."""
    fresh = type(settings)(_env_file=None)
    assert fresh.ai_3d_require_external_consent is True


# ── B) Provider safety ──────────────────────────────────────────────────────

def test_rodin_blocked_when_global_switch_off(monkeypatch):
    """Rodin is unavailable when global remote switch is False (default)."""
    monkeypatch.setattr(settings, "ai_3d_remote_providers_enabled", False)
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "some_key")
    monkeypatch.setattr(settings, "rodin_mock_mode", True)

    provider = RodinProvider()
    avail, reason = provider.is_available()
    assert avail is False
    assert reason == "remote_providers_disabled_globally"


def test_rodin_disabled_returns_unavailable(monkeypatch):
    """Rodin returns unavailable when provider-level switch disabled."""
    monkeypatch.setattr(settings, "ai_3d_remote_providers_enabled", True)
    monkeypatch.setattr(settings, "rodin_enabled", False)

    provider = RodinProvider()
    avail, reason = provider.is_available()
    assert avail is False
    assert "disabled" in reason.lower()


def test_rodin_missing_key_returns_unavailable(monkeypatch):
    """Rodin returns unavailable when API key is missing."""
    monkeypatch.setattr(settings, "ai_3d_remote_providers_enabled", True)
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "")
    monkeypatch.setattr(settings, "rodin_mock_mode", True)
    # Provide valid env so mock mode doesn't block
    from modules.operations.settings import AppEnvironment
    monkeypatch.setattr(settings, "env", AppEnvironment.LOCAL_DEV)

    provider = RodinProvider()
    avail, reason = provider.is_available()
    assert avail is False
    assert "API key is missing" in reason


def test_rodin_mock_mode_false_returns_not_implemented(monkeypatch):
    """Rodin returns not_implemented when mock mode is false and no real API."""
    monkeypatch.setattr(settings, "ai_3d_remote_providers_enabled", True)
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "test_key")
    monkeypatch.setattr(settings, "rodin_mock_mode", False)

    provider = RodinProvider()
    avail, reason = provider.is_available()
    assert avail is False
    assert reason == "rodin_real_api_not_implemented"


def test_rodin_mock_mode_prohibited_in_production(monkeypatch):
    """Rodin mock mode is prohibited in non-local_dev environments."""
    monkeypatch.setattr(settings, "ai_3d_remote_providers_enabled", True)
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "test_key")
    monkeypatch.setattr(settings, "rodin_mock_mode", True)
    monkeypatch.setattr(settings, "env", "production")

    provider = RodinProvider()
    avail, reason = provider.is_available()
    assert avail is False
    assert "prohibited" in reason.lower()


# ── C) Consent gate ─────────────────────────────────────────────────────────

def test_rodin_with_consent_mock_remote_switch_can_proceed(tmp_path, monkeypatch):
    """Rodin with consent + mock enabled + remote switch enabled can proceed."""
    from modules.operations.settings import AppEnvironment
    monkeypatch.setattr(settings, "ai_3d_remote_providers_enabled", True)
    monkeypatch.setattr(settings, "env", AppEnvironment.LOCAL_DEV)
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "mock_key")
    monkeypatch.setattr(settings, "rodin_mock_mode", True)

    provider = RodinProvider()
    avail, _ = provider.is_available()
    assert avail is True

    input_img = tmp_path / "input.jpg"
    from PIL import Image
    Image.new("RGB", (1, 1)).save(input_img, "JPEG")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = provider.generate(str(input_img), str(output_dir), {"external_provider_consent": True})
    assert result["status"] == "ok"
    assert result["provider"] == "rodin"
    assert result["metadata"]["external_provider"] is True


def test_sf3d_does_not_require_external_consent():
    """SF3DProvider.name is sf3d, which is not in the external_providers set."""
    from modules.ai_3d_generation.sf3d_provider import SF3DProvider
    p = SF3DProvider()
    assert p.name == "sf3d"
    assert p.name not in {"rodin", "meshy", "tripo"}


# ── B) Provider registry / unknown provider ──────────────────────────────────

def test_unknown_provider_raises_value_error():
    """_get_provider raises ValueError for unknown provider names — no SF3D fallback."""
    from modules.ai_3d_generation.pipeline import _get_provider
    with pytest.raises(ValueError, match="unknown_ai3d_provider:banana"):
        _get_provider("banana")


def test_sf3d_provider_resolves():
    """sf3d resolves to SF3DProvider."""
    from modules.ai_3d_generation.pipeline import _get_provider
    from modules.ai_3d_generation.sf3d_provider import SF3DProvider
    p = _get_provider("sf3d")
    assert isinstance(p, SF3DProvider)


def test_rodin_provider_resolves():
    """rodin resolves to RodinProvider."""
    from modules.ai_3d_generation.pipeline import _get_provider
    p = _get_provider("rodin")
    assert isinstance(p, RodinProvider)


# ── D) Sanitization ─────────────────────────────────────────────────────────

def test_sanitize_json_like_removes_bearer_from_list():
    """sanitize_json_like redacts Bearer tokens from list elements."""
    from modules.ai_3d_generation.sanitization import sanitize_json_like
    obj = ["Authorization: Bearer SECRET_TOKEN", "normal text"]
    result = sanitize_json_like(obj)
    assert "SECRET_TOKEN" not in result[0]
    assert "[REDACTED]" in result[0]
    assert result[1] == "normal text"


def test_sanitize_json_like_removes_api_key_from_dict():
    """sanitize_json_like redacts api_key from nested dict."""
    from modules.ai_3d_generation.sanitization import sanitize_json_like
    obj = {"provider_failure_reason": "api_key=MY_API_KEY", "ok": True}
    result = sanitize_json_like(obj)
    assert "MY_API_KEY" not in result["provider_failure_reason"]
    assert result["ok"] is True


def test_sanitize_json_like_nested():
    """sanitize_json_like handles nested dicts and lists."""
    from modules.ai_3d_generation.sanitization import sanitize_json_like
    obj = {
        "warnings": ["Bearer SECRET123", "ok"],
        "metadata": {"provider_error": "token=SECRET456"},
        "count": 5,
    }
    result = sanitize_json_like(obj)
    assert "SECRET123" not in str(result["warnings"])
    assert "SECRET456" not in str(result["metadata"])
    assert result["count"] == 5


def test_rodin_unavailable_no_secret_in_safe_generate(monkeypatch):
    """Unavailable reason does not leak secret in safe_generate result."""
    monkeypatch.setattr(settings, "rodin_api_key", "MY_SECRET_KEY")
    monkeypatch.setattr(settings, "rodin_enabled", False)

    provider = RodinProvider()
    # Override is_available to simulate a leaky reason
    monkeypatch.setattr(provider, "is_available", lambda: (False, "Failed with MY_SECRET_KEY"))

    result = provider.safe_generate("input.jpg", "output_dir")
    assert "MY_SECRET_KEY" not in result["error"]
    assert "MY_SECRET_KEY" not in result["metadata"].get("sanitized_error", "")
    assert "[REDACTED]" in result["error"]


# ── Legacy lifecycle test (kept with global switch enabled) ──────────────────

def test_rodin_provider_lifecycle_mocked(tmp_path, monkeypatch):
    """Verify the mocked lifecycle of RodinProvider."""
    from modules.operations.settings import AppEnvironment
    monkeypatch.setattr(settings, "ai_3d_remote_providers_enabled", True)
    monkeypatch.setattr(settings, "env", AppEnvironment.LOCAL_DEV)
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "mock_key")
    monkeypatch.setattr(settings, "rodin_mock_mode", True)

    provider = RodinProvider()

    input_img = tmp_path / "input.jpg"
    from PIL import Image
    Image.new("RGB", (1, 1)).save(input_img, "JPEG")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = provider.generate(str(input_img), str(output_dir), {"external_provider_consent": True})

    assert result["status"] == "ok"
    assert result["provider"] == "rodin"
    assert "mock_rodin_task_123" in result["output_path"]
    assert result["metadata"]["external_task_id"] == "mock_rodin_task_123"
    assert result["metadata"]["external_status"] == "succeeded"
    assert result["metadata"]["external_provider"] is True


def test_rodin_unavailable_metadata(monkeypatch):
    """Unavailable results contain required Phase 1.5 metadata."""
    monkeypatch.setattr(settings, "ai_3d_remote_providers_enabled", False)
    monkeypatch.setattr(settings, "rodin_enabled", False)
    provider = RodinProvider()

    result = provider.safe_generate("input.jpg", "output_dir")

    assert result["status"] == "unavailable"
    assert result["metadata"]["external_provider"] is True
    assert result["metadata"]["external_provider_name"] == "rodin"
    assert result["metadata"]["external_status"] == "unavailable"
    assert result["metadata"]["provider_poll_count"] == 0
    assert "provider_latency_sec" in result["metadata"]
    assert "privacy_notice" in result["metadata"]
    assert "sanitized_error" in result["metadata"]
    assert result["metadata"]["sanitized_error"] == result["error"]


# ── E) Manifest recursive sanitization ──────────────────────────────────────

def _base_manifest_kwargs(**overrides):
    """Minimal valid kwargs for build_manifest() for test convenience."""
    from modules.ai_3d_generation.manifest import build_manifest
    defaults = dict(
        session_id="s1",
        source_input_path="/in/img.jpg",
        input_type="image",
        provider="rodin",
        provider_status="ok",
        model_name="gen-2",
        license_note="test",
        selected_frame_path=None,
        prepared_image_path=None,
        preprocessing={},
        postprocessing={},
        quality_gate={},
        output_glb_path="out.glb",
        output_format="glb",
        preview_image_path=None,
        status="ok",
        warnings=[],
        errors=[],
    )
    defaults.update(overrides)
    return build_manifest(**defaults)


def test_manifest_sanitizes_warnings():
    """Bearer token in warnings[] is redacted in manifest."""
    import json
    m = _base_manifest_kwargs(warnings=["Authorization: Bearer SECRET_WARN"])
    assert "SECRET_WARN" not in json.dumps(m)
    assert "[REDACTED]" in json.dumps(m["warnings"])


def test_manifest_sanitizes_errors():
    """api_key in errors[] is redacted in manifest."""
    import json
    m = _base_manifest_kwargs(errors=["api_key=SECRET_ERR"])
    assert "SECRET_ERR" not in json.dumps(m)


def test_manifest_sanitizes_worker_metadata():
    """Bearer token in worker_metadata is redacted in manifest."""
    import json
    m = _base_manifest_kwargs(
        worker_metadata={"log": "Authorization: Bearer SECRET_WORKER"}
    )
    assert "SECRET_WORKER" not in json.dumps(m)


def test_manifest_sanitizes_candidates():
    """Bearer token nested inside candidates[] is redacted in manifest."""
    import json
    m = _base_manifest_kwargs(candidates=[
        {"candidate_id": "cand_001", "error": "token=SECRET_CAND"}
    ])
    assert "SECRET_CAND" not in json.dumps(m)


def test_manifest_sanitizes_candidate_ranking():
    """Bearer token in candidate_ranking[] is redacted in manifest."""
    import json
    m = _base_manifest_kwargs(candidate_ranking=[
        {"candidate_id": "cand_001", "note": "Bearer SECRET_RANK"}
    ])
    assert "SECRET_RANK" not in json.dumps(m)


def test_manifest_sanitizes_path_diagnostics():
    """token= in path_diagnostics is redacted in manifest."""
    import json
    m = _base_manifest_kwargs(
        path_diagnostics={"debug": "token=SECRET_DIAG"}
    )
    assert "SECRET_DIAG" not in json.dumps(m)


def test_manifest_sanitizes_provider_failure_reason():
    """api_key in provider_failure_reason is redacted in manifest."""
    import json
    m = _base_manifest_kwargs(
        provider_failure_reason="api_key=SECRET_FAIL",
        status="failed",
        provider_status="failed",
    )
    assert "SECRET_FAIL" not in json.dumps(m)


# ── F) Remote provider exception metadata ───────────────────────────────────

def test_safe_generate_exception_has_external_metadata(monkeypatch):
    """When generate() raises, safe_generate must still include external metadata."""
    from modules.operations.settings import AppEnvironment
    monkeypatch.setattr(settings, "ai_3d_remote_providers_enabled", True)
    monkeypatch.setattr(settings, "env", AppEnvironment.LOCAL_DEV)
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "mock_key")
    monkeypatch.setattr(settings, "rodin_mock_mode", True)

    provider = RodinProvider()

    # Override generate() to raise with a leaky message
    def _raise(*a, **kw):
        raise RuntimeError("Auth failed: Bearer SECRET_EXCEPTION_TOKEN")

    monkeypatch.setattr(provider, "generate", _raise)

    result = provider.safe_generate("input.jpg", "out_dir")

    assert result["status"] == "failed"
    assert result["metadata"]["external_provider"] is True
    assert result["metadata"]["external_provider_name"] == "rodin"
    assert result["metadata"]["external_status"] == "exception"
    assert "provider_latency_sec" in result["metadata"]
    assert "provider_poll_count" in result["metadata"]
    assert "privacy_notice" in result["metadata"]
    assert "sanitized_error" in result["metadata"]


def test_safe_generate_exception_no_secret_leak(monkeypatch):
    """Exception text containing a secret must be sanitized in safe_generate result."""
    import json
    from modules.operations.settings import AppEnvironment
    monkeypatch.setattr(settings, "ai_3d_remote_providers_enabled", True)
    monkeypatch.setattr(settings, "env", AppEnvironment.LOCAL_DEV)
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "mock_key")
    monkeypatch.setattr(settings, "rodin_mock_mode", True)

    provider = RodinProvider()

    def _raise(*a, **kw):
        raise RuntimeError("api_key=MY_SUPER_SECRET_KEY")

    monkeypatch.setattr(provider, "generate", _raise)

    result = provider.safe_generate("input.jpg", "out_dir")
    serialized = json.dumps(result)

    assert "MY_SUPER_SECRET_KEY" not in serialized
    assert "[REDACTED]" in serialized


def test_safe_generate_create_task_exception_metadata(monkeypatch):
    """When create_task raises inside generate(), safe_generate catches it with full metadata."""
    from modules.operations.settings import AppEnvironment
    monkeypatch.setattr(settings, "ai_3d_remote_providers_enabled", True)
    monkeypatch.setattr(settings, "env", AppEnvironment.LOCAL_DEV)
    monkeypatch.setattr(settings, "rodin_enabled", True)
    monkeypatch.setattr(settings, "rodin_api_key", "mock_key")
    monkeypatch.setattr(settings, "rodin_mock_mode", True)

    provider = RodinProvider()

    def _bad_create(*a, **kw):
        raise ConnectionError("Bearer SECRET_CONN_TOKEN network failed")

    monkeypatch.setattr(provider, "create_task", _bad_create)

    result = provider.safe_generate("input.jpg", "out_dir")

    assert result["status"] == "failed"
    assert result["metadata"].get("external_provider") is True
    assert result["metadata"].get("external_status") == "exception"
    assert "SECRET_CONN_TOKEN" not in str(result)
