"""
Tests for Depth Studio API hardening:
  - provider unavailable → enriched process response body
  - missing_outputs list is accurate
  - _depth_session_summary helper
  - _artifact_404 detail structure
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    from modules.operations.api import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def api_headers():
    from modules.operations.settings import settings
    return {"X-API-Key": settings.pilot_api_key or "test-key"}


def _register_session(session_id: str, provider: str = "depth_anything_v2",
                       status: str = "uploaded", input_path: str = "/tmp/fake.jpg",
                       manifest_path: str | None = None):
    """Directly inject a session into the in-memory dict (no file I/O needed)."""
    from modules.operations.api import _depth_sessions
    _depth_sessions[session_id] = {
        "session_id": session_id,
        "status": status,
        "input_path": input_path,
        "provider": provider,
        "manifest_path": manifest_path,
    }


# ── process response body tests ───────────────────────────────────────────────

class TestProcessResponseBody:
    def test_failed_manifest_returns_enriched_body(self, client, api_headers, tmp_path):
        """When pipeline returns status=failed, the /process response must include
        provider_status, warnings, and missing_outputs — not just manifest."""
        session_id = f"ds_{uuid.uuid4().hex[:8]}"
        _register_session(session_id, provider="depth_pro",
                          input_path=str(tmp_path / "img.jpg"))

        failed_manifest = {
            "status": "failed",
            "provider": "depth_pro",
            "provider_status": "unavailable",
            "warnings": ["experimental_provider", "depth_pro_disabled"],
            "glb_path": None,
        }

        with patch("modules.depth_studio.pipeline.run_depth_studio",
                   return_value=failed_manifest):
            r = client.post(
                f"/api/depth-studio/process/{session_id}",
                headers=api_headers,
            )

        assert r.status_code == 200          # HTTP stays 200 per spec
        body = r.json()
        assert body["status"] == "failed"
        assert body["provider_status"] == "unavailable"
        assert isinstance(body["warnings"], list)
        assert isinstance(body["missing_outputs"], list)
        # All five artifacts should be missing (nothing written to disk)
        for key in ("depth_preview", "subject_mask", "mask_overlay",
                    "cropped_subject", "glb"):
            assert key in body["missing_outputs"], (
                f"Expected '{key}' in missing_outputs, got {body['missing_outputs']}"
            )

    def test_depth_pro_failure_includes_reason(self, client, api_headers, tmp_path):
        """Depth Pro unavailable → provider_failure_reason in response body."""
        session_id = f"ds_{uuid.uuid4().hex[:8]}"
        _register_session(session_id, provider="depth_pro",
                          input_path=str(tmp_path / "img.jpg"))

        failed_manifest = {
            "status": "failed",
            "provider": "depth_pro",
            "provider_status": "unavailable",
            "warnings": [],
        }

        with patch("modules.depth_studio.pipeline.run_depth_studio",
                   return_value=failed_manifest):
            r = client.post(
                f"/api/depth-studio/process/{session_id}",
                headers=api_headers,
            )

        body = r.json()
        assert "provider_failure_reason" in body
        assert "DEPTH_PRO_ENABLED" in body["provider_failure_reason"]

    def test_ok_manifest_no_provider_failure_reason(self, client, api_headers, tmp_path):
        """Successful run must NOT include provider_failure_reason."""
        session_id = f"ds_{uuid.uuid4().hex[:8]}"
        _register_session(session_id, input_path=str(tmp_path / "img.jpg"))

        ok_manifest = {
            "status": "ok",
            "provider": "depth_anything_v2",
            "provider_status": "ok",
            "warnings": [],
            "glb_path": "/some/path.glb",
        }

        with patch("modules.depth_studio.pipeline.run_depth_studio",
                   return_value=ok_manifest):
            r = client.post(
                f"/api/depth-studio/process/{session_id}",
                headers=api_headers,
            )

        body = r.json()
        assert body["status"] == "ok"
        assert "provider_failure_reason" not in body


# ── artifact 404 enrichment tests ─────────────────────────────────────────────

class TestArtifact404Detail:
    def _check_404_detail(self, detail: dict):
        assert "session_status" in detail, f"missing session_status in {detail}"
        assert "provider_status" in detail, f"missing provider_status in {detail}"
        assert "warnings" in detail
        assert "missing_outputs" in detail

    def test_preview_404_has_session_summary(self, client, api_headers, tmp_path):
        session_id = f"ds_{uuid.uuid4().hex[:8]}"
        manifest = {
            "status": "failed",
            "provider": "depth_pro",
            "provider_status": "unavailable",
            "warnings": ["depth_pro_disabled"],
        }
        manifest_path = tmp_path / "depth_studio_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        _register_session(session_id, provider="depth_pro",
                          status="failed",
                          manifest_path=str(manifest_path))

        r = client.get(f"/api/depth-studio/preview/{session_id}", headers=api_headers)
        assert r.status_code == 404
        detail = r.json()["detail"]
        self._check_404_detail(detail)
        assert detail["provider_status"] == "unavailable"
        assert "depth_pro_disabled" in detail["warnings"]
        assert "provider_failure_reason" in detail
        assert "DEPTH_PRO_ENABLED" in detail["provider_failure_reason"]

    def test_mask_overlay_404_has_session_summary(self, client, api_headers, tmp_path):
        session_id = f"ds_{uuid.uuid4().hex[:8]}"
        _register_session(session_id, provider="depth_anything_v2",
                          status="failed")

        r = client.get(f"/api/depth-studio/mask-overlay/{session_id}", headers=api_headers)
        assert r.status_code == 404
        detail = r.json()["detail"]
        self._check_404_detail(detail)

    def test_subject_mask_404_has_session_summary(self, client, api_headers):
        session_id = f"ds_{uuid.uuid4().hex[:8]}"
        _register_session(session_id)

        r = client.get(f"/api/depth-studio/subject-mask/{session_id}", headers=api_headers)
        assert r.status_code == 404
        self._check_404_detail(r.json()["detail"])

    def test_cropped_subject_404_has_session_summary(self, client, api_headers):
        session_id = f"ds_{uuid.uuid4().hex[:8]}"
        _register_session(session_id)

        r = client.get(f"/api/depth-studio/cropped-subject/{session_id}", headers=api_headers)
        assert r.status_code == 404
        self._check_404_detail(r.json()["detail"])


# ── _depth_session_summary unit tests ─────────────────────────────────────────

class TestDepthSessionSummary:
    def test_depth_pro_unavailable_reason(self, tmp_path):
        from modules.operations.api import _depth_sessions, _depth_session_summary

        session_id = f"ds_{uuid.uuid4().hex[:8]}"
        manifest = {
            "status": "failed",
            "provider": "depth_pro",
            "provider_status": "unavailable",
            "warnings": ["depth_pro_disabled"],
        }
        mp = tmp_path / "depth_studio_manifest.json"
        mp.write_text(json.dumps(manifest), encoding="utf-8")
        _depth_sessions[session_id] = {
            "status": "failed",
            "provider": "depth_pro",
            "manifest_path": str(mp),
        }

        summary = _depth_session_summary(session_id)
        assert summary["provider_status"] == "unavailable"
        assert "provider_failure_reason" in summary
        assert "DEPTH_PRO_ENABLED" in summary["provider_failure_reason"]

    def test_missing_outputs_all_absent(self, tmp_path):
        from modules.operations.api import _depth_sessions, _depth_session_summary

        session_id = f"ds_{uuid.uuid4().hex[:8]}"
        _depth_sessions[session_id] = {
            "status": "failed",
            "provider": "depth_anything_v2",
            "manifest_path": None,
        }

        summary = _depth_session_summary(session_id)
        for key in ("depth_preview", "subject_mask", "mask_overlay",
                    "cropped_subject", "glb"):
            assert key in summary["missing_outputs"]

    def test_no_provider_failure_reason_for_ok_provider(self, tmp_path):
        from modules.operations.api import _depth_sessions, _depth_session_summary

        session_id = f"ds_{uuid.uuid4().hex[:8]}"
        manifest = {
            "status": "ok",
            "provider": "depth_anything_v2",
            "provider_status": "ok",
            "warnings": [],
        }
        mp = tmp_path / "manifest.json"
        mp.write_text(json.dumps(manifest), encoding="utf-8")
        _depth_sessions[session_id] = {
            "status": "ok",
            "provider": "depth_anything_v2",
            "manifest_path": str(mp),
        }

        summary = _depth_session_summary(session_id)
        assert "provider_failure_reason" not in summary

    def test_depth_pro_error_status_reason(self, tmp_path):
        """provider_status='error' (worker crash) must produce provider_failure_reason."""
        from modules.operations.api import _depth_sessions, _depth_session_summary

        session_id = f"ds_{uuid.uuid4().hex[:8]}"
        manifest = {
            "status": "failed",
            "provider": "depth_pro",
            "provider_status": "failed",   # normalised from "error" by safe_infer
            "warnings": [],
        }
        mp = tmp_path / "manifest.json"
        mp.write_text(json.dumps(manifest), encoding="utf-8")
        _depth_sessions[session_id] = {
            "status": "failed",
            "provider": "depth_pro",
            "manifest_path": str(mp),
        }

        summary = _depth_session_summary(session_id)
        assert "provider_failure_reason" in summary
        # "failed" for depth_pro → "unavailable" message path
        assert "Depth Pro" in summary["provider_failure_reason"]

    def test_generic_provider_failure_reason(self, tmp_path):
        from modules.operations.api import _depth_sessions, _depth_session_summary

        session_id = f"ds_{uuid.uuid4().hex[:8]}"
        manifest = {
            "status": "failed",
            "provider": "depth_anything_v2",
            "provider_status": "failed",
            "warnings": [],
        }
        mp = tmp_path / "manifest.json"
        mp.write_text(json.dumps(manifest), encoding="utf-8")
        _depth_sessions[session_id] = {
            "status": "failed",
            "provider": "depth_anything_v2",
            "manifest_path": str(mp),
        }

        summary = _depth_session_summary(session_id)
        assert "provider_failure_reason" in summary
        assert "depth_anything_v2" in summary["provider_failure_reason"]


# ── safe_infer normalisation tests ───────────────────────────────────────────

class TestSafeInferNormalisation:
    """safe_infer must normalise non-ok, non-standard status values to 'failed'."""

    def _make_provider(self, infer_return: dict):
        from modules.depth_studio.depth_provider_base import DepthProviderBase

        class _FakeProvider(DepthProviderBase):
            name = "fake"
            license_note = ""
            def is_available(self):
                return True, ""
            def infer(self, image_path, output_dir):
                return infer_return

        return _FakeProvider()

    def test_error_status_normalised_to_failed(self):
        p = self._make_provider({"status": "error", "message": "worker crashed"})
        result = p.safe_infer("/fake/img.jpg", "/fake/out")
        assert result["status"] == "failed", f"Expected 'failed', got '{result['status']}'"
        assert result.get("reason")   # original message preserved

    def test_ok_status_passes_through(self):
        p = self._make_provider({
            "status": "ok", "depth_map_path": "/out/depth.png",
            "depth_format": "png16", "model_name": "test", "warnings": []
        })
        result = p.safe_infer("/fake/img.jpg", "/fake/out")
        assert result["status"] == "ok"

    def test_unavailable_status_passes_through(self):
        p = self._make_provider({"status": "unavailable", "reason": "disabled"})
        result = p.safe_infer("/fake/img.jpg", "/fake/out")
        assert result["status"] == "unavailable"

    def test_arbitrary_unknown_status_normalised(self):
        p = self._make_provider({"status": "timeout", "reason": "took too long"})
        result = p.safe_infer("/fake/img.jpg", "/fake/out")
        assert result["status"] == "failed"
        assert result.get("reason") is not None  # original reason preserved
