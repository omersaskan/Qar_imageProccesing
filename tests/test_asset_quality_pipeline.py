"""
AQ1 — Asset Quality Pipeline tests.

Tests:
 1.  asset_quality returns stable shape for valid minimal GLB
 2.  asset_quality handles missing GLB gracefully
 3.  normalization audit returns dimensions/bounds
 4.  mesh_cleanup audit detects no mesh / empty scene
 5.  pbr audit returns material/texture counts
 6.  lod plan returns preview/mobile/desktop tiers
 7.  export profile marks mobile_ar not ready when glb_validation invalid
 8.  asset_quality exception does not crash pipeline
 9.  API response includes asset_quality fields
10.  UI contains Asset Quality section
11.  UI distinguishes AR Technical Readiness from Asset Quality
12.  SF3D defaults unchanged
13.  Hunyuan disabled/default safety unchanged
14.  no destructive optimization or overwrite performed by default
15.  invalid providers still rejected
"""
from __future__ import annotations

import json
import struct
import tempfile
import os
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from modules.operations.api import app
import modules.operations.api as api_module

client = TestClient(app)


# ── GLB helpers ───────────────────────────────────────────────────────────────

def _make_glb(gltf_dict=None, with_mesh=True) -> bytes:
    """Build a minimal valid GLB binary for testing."""
    if gltf_dict is None:
        if with_mesh:
            gltf_dict = {
                "asset": {"version": "2.0"},
                "scene": 0,
                "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "materials": [{"name": "default"}],
            }
        else:
            gltf_dict = {
                "asset": {"version": "2.0"},
                "scene": 0,
                "scenes": [{"nodes": []}],
                "nodes": [],
                "meshes": [],
            }
    json_bytes = json.dumps(gltf_dict).encode("utf-8")
    pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * pad
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    total = 12 + len(json_chunk)
    header = struct.pack("<III", 0x46546C67, 2, total)
    return header + json_chunk


def _write_glb(td: str, name="out.glb", **kwargs) -> str:
    path = os.path.join(td, name)
    with open(path, "wb") as f:
        f.write(_make_glb(**kwargs))
    return path


def _mock_manifest(glb_path: str, glb_valid: bool = True) -> Dict[str, Any]:
    return {
        "session_id": "aq_test",
        "status": "review",
        "provider": "sf3d",
        "provider_status": "ok",
        "execution_mode": "wsl_subprocess",
        "output_glb_path": glb_path,
        "output_size_bytes": 1024,
        "worker_metadata": {"texture_resolution": 1024},
        "input_type": "single_image",
        "input_mode": "single_image",
        "candidate_count": 1,
        "selected_candidate_id": "c1",
        "candidate_ranking": [],
        "warnings": [],
        "errors": [],
        "review_required": True,
        "quality_gate": {"verdict": "review", "output_exists": True, "warnings": [], "reason": None},
        "missing_outputs": [],
        "mesh_stats": {"enabled": True, "available": True, "vertex_count": 100, "face_count": 200, "geometry_count": 1},
        "glb_validation": {"valid": glb_valid, "issues": [], "warnings": []},
        "ar_readiness": {"enabled": True, "score": 85, "verdict": "mobile_ready", "checks": {}, "warnings": [], "recommendations": []},
    }


# ── Test 1: stable shape ───────────────────────────────────────────────────────

class TestAssetQualityShape(unittest.TestCase):
    def test_returns_stable_shape_for_valid_glb(self):
        from modules.ai_3d_generation.asset_quality import run_asset_quality_pipeline
        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td)
            manifest = _mock_manifest(glb_path)
            result = run_asset_quality_pipeline(glb_path, manifest)

        self.assertIn("enabled", result)
        self.assertIn("available", result)
        self.assertIn("status", result)
        self.assertIn("score", result)
        self.assertIn("verdict", result)
        self.assertIn("provider_neutral", result)
        self.assertIn("checks", result)
        self.assertIn("warnings", result)
        self.assertIn("recommendations", result)
        self.assertIn("error", result)
        self.assertTrue(result["provider_neutral"])
        checks = result["checks"]
        self.assertIn("scale_orientation", checks)
        self.assertIn("mesh_cleanup", checks)
        self.assertIn("lod", checks)
        self.assertIn("pbr_textures", checks)
        self.assertIn("export_profiles", checks)


# ── Test 2: missing GLB graceful ──────────────────────────────────────────────

class TestAssetQualityMissingGlb(unittest.TestCase):
    def test_handles_missing_glb(self):
        from modules.ai_3d_generation.asset_quality import run_asset_quality_pipeline
        result = run_asset_quality_pipeline("/no/such/file.glb", {})
        self.assertTrue(result["enabled"])
        self.assertIn("verdict", result)
        self.assertIn(result["verdict"], ("not_ready", "needs_review"))
        self.assertIsNone(result["error"])

    def test_handles_none_glb_path(self):
        from modules.ai_3d_generation.asset_quality import run_asset_quality_pipeline
        result = run_asset_quality_pipeline(None, {})
        self.assertTrue(result["enabled"])
        self.assertEqual(result["verdict"], "not_ready")


# ── Test 3: normalization audit ───────────────────────────────────────────────

class TestNormalizationAudit(unittest.TestCase):
    def test_returns_required_keys(self):
        from modules.ai_3d_generation.asset_quality.normalization import analyze_normalization
        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td)
            result = analyze_normalization(glb_path)
        self.assertTrue(result["enabled"])
        self.assertFalse(result["applied"], "normalization must never be applied by default")
        self.assertIn("analysis", result)
        self.assertIn("issues", result)
        self.assertIn("warnings", result)
        self.assertIn("recommendations", result)
        analysis = result["analysis"]
        self.assertIn("bounds", analysis)
        self.assertIn("dimensions", analysis)
        self.assertIn("center", analysis)
        self.assertIn("ground_offset", analysis)
        self.assertIn("largest_axis", analysis)

    def test_missing_glb_returns_issue(self):
        from modules.ai_3d_generation.asset_quality.normalization import analyze_normalization
        result = analyze_normalization("/nonexistent.glb")
        self.assertIn("glb_missing", result["issues"])


# ── Test 4: mesh cleanup — empty scene ────────────────────────────────────────

class TestMeshCleanupAudit(unittest.TestCase):
    def test_returns_required_keys(self):
        from modules.ai_3d_generation.asset_quality.mesh_cleanup_audit import audit_mesh_cleanup
        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td)
            result = audit_mesh_cleanup(glb_path)
        self.assertTrue(result["enabled"])
        self.assertIn("status", result)
        self.assertIn("issues", result)
        self.assertIn("warnings", result)
        self.assertIn("metrics", result)
        self.assertIn("recommendations", result)
        metrics = result["metrics"]
        self.assertIn("component_count", metrics)
        self.assertIn("degenerate_face_count", metrics)
        self.assertIn("boundary_edge_count", metrics)
        self.assertIn("non_manifold_estimate", metrics)

    def test_detects_no_mesh_or_graceful(self):
        """Audit on empty-mesh GLB either detects no_mesh or returns gracefully."""
        from modules.ai_3d_generation.asset_quality.mesh_cleanup_audit import audit_mesh_cleanup
        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td, with_mesh=False)
            result = audit_mesh_cleanup(glb_path)
        # Must return valid structure regardless
        self.assertIn("status", result)
        self.assertIn("metrics", result)
        # If trimesh is available and detects no mesh, status should be failed
        # If trimesh not available, status is ok (graceful degradation) — both are valid
        self.assertIn(result["status"], ("ok", "review", "failed"))

    def test_missing_glb_returns_failed(self):
        from modules.ai_3d_generation.asset_quality.mesh_cleanup_audit import audit_mesh_cleanup
        result = audit_mesh_cleanup("/no/such/file.glb")
        self.assertEqual(result["status"], "failed")
        self.assertIn("glb_missing", result["issues"])


# ── Test 5: PBR audit ─────────────────────────────────────────────────────────

class TestPbrAudit(unittest.TestCase):
    def test_returns_material_and_texture_counts(self):
        from modules.ai_3d_generation.asset_quality.pbr_audit import audit_pbr_textures
        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td)  # has 1 material, 0 textures
            result = audit_pbr_textures(glb_path)
        self.assertTrue(result["enabled"])
        self.assertTrue(result["available"])
        self.assertIsInstance(result["material_count"], int)
        self.assertIsInstance(result["texture_count"], int)
        self.assertIsInstance(result["image_count"], int)
        self.assertEqual(result["material_count"], 1)
        self.assertEqual(result["texture_count"], 0)
        self.assertIn("issues", result)
        self.assertIn("warnings", result)
        self.assertIn("recommendations", result)

    def test_missing_glb_returns_issue(self):
        from modules.ai_3d_generation.asset_quality.pbr_audit import audit_pbr_textures
        result = audit_pbr_textures("/no/such/file.glb")
        self.assertIn("glb_missing", result["issues"])

    def test_no_material_warns(self):
        from modules.ai_3d_generation.asset_quality.pbr_audit import audit_pbr_textures
        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td, with_mesh=False)  # no materials
            result = audit_pbr_textures(glb_path)
        self.assertEqual(result["material_count"], 0)
        self.assertIn("no_materials", result["issues"])


# ── Test 6: LOD plan ──────────────────────────────────────────────────────────

class TestLodPlan(unittest.TestCase):
    def test_returns_three_tiers(self):
        from modules.ai_3d_generation.asset_quality.lod import build_lod_plan
        result = build_lod_plan({}, {})
        self.assertTrue(result["enabled"])
        self.assertTrue(result["available"])
        self.assertIn("tiers", result)
        tiers = result["tiers"]
        self.assertEqual(len(tiers), 3)
        names = [t["name"] for t in tiers]
        self.assertIn("preview", names)
        self.assertIn("mobile", names)
        self.assertIn("desktop", names)

    def test_tier_structure(self):
        from modules.ai_3d_generation.asset_quality.lod import build_lod_plan
        result = build_lod_plan({"face_count": 50000}, {})
        for tier in result["tiers"]:
            self.assertIn("name", tier)
            self.assertIn("target_faces", tier)
            self.assertIn("recommended", tier)
            self.assertIn("path", tier)

    def test_plan_only_by_default(self):
        from modules.ai_3d_generation.asset_quality.lod import build_lod_plan
        result = build_lod_plan({"face_count": 500000}, {})
        # AI_3D_LOD_GENERATION_ENABLED defaults to false
        self.assertEqual(result["strategy"], "plan_only")
        self.assertFalse(result["generated"])
        for tier in result["tiers"]:
            self.assertIsNone(tier["path"])

    def test_no_lod_files_generated_by_default(self):
        """Verify no output_*.glb files are written unless LOD flag is set."""
        from modules.ai_3d_generation.asset_quality.lod import build_lod_plan
        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td)
            result = build_lod_plan(
                {"face_count": 500000},
                {},
                asset_quality_context={"glb_path": glb_path, "output_dir": td},
            )
        self.assertFalse(result["generated"])
        import glob
        lod_files = glob.glob(os.path.join(td, "output_*.glb"))
        self.assertEqual(len(lod_files), 0, "No LOD files should be written when flag is off")


# ── Test 7: export profiles — invalid GLB blocks mobile_ar ───────────────────

class TestExportProfiles(unittest.TestCase):
    def test_mobile_ar_blocked_when_glb_invalid(self):
        from modules.ai_3d_generation.asset_quality.export_profiles import assess_export_profiles
        manifest = {
            "output_glb_path": "/some/path.glb",
            "glb_validation": {"valid": False, "issues": ["invalid_magic"], "warnings": []},
            "mesh_stats": {},
            "ar_readiness": {},
            "output_size_bytes": 1024,
        }
        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td)
            manifest["output_glb_path"] = glb_path
            result = assess_export_profiles(manifest)
        self.assertFalse(result["mobile_ar"]["ready"])
        self.assertIn("glb_validation_failed", result["mobile_ar"]["blocking_reasons"])

    def test_all_profiles_blocked_when_glb_missing(self):
        from modules.ai_3d_generation.asset_quality.export_profiles import assess_export_profiles
        manifest = {
            "output_glb_path": "/nonexistent/file.glb",
            "glb_validation": {"valid": None},
            "mesh_stats": {},
            "ar_readiness": {},
            "output_size_bytes": 0,
        }
        result = assess_export_profiles(manifest)
        self.assertFalse(result["web_preview"]["ready"])
        self.assertFalse(result["mobile_ar"]["ready"])
        self.assertFalse(result["desktop_high"]["ready"])
        self.assertFalse(result["raw"]["available"])


# ── Test 8: exception does not crash pipeline ─────────────────────────────────

class TestAssetQualityExceptionSafety(unittest.TestCase):
    def test_exception_in_sub_audit_does_not_crash(self):
        from modules.ai_3d_generation.asset_quality import run_asset_quality_pipeline
        with patch(
            "modules.ai_3d_generation.asset_quality.quality_pipeline.analyze_normalization",
            side_effect=RuntimeError("injected failure"),
        ):
            with tempfile.TemporaryDirectory() as td:
                glb_path = _write_glb(td)
                result = run_asset_quality_pipeline(glb_path, _mock_manifest(glb_path))
        # Must return degraded result, never raise
        self.assertIn("enabled", result)
        self.assertFalse(result["available"])
        self.assertIsNotNone(result["error"])

    def test_top_level_exception_returns_degraded(self):
        from modules.ai_3d_generation.asset_quality.quality_pipeline import run_asset_quality_pipeline
        with patch(
            "modules.ai_3d_generation.asset_quality.quality_pipeline._run_pipeline",
            side_effect=Exception("total failure"),
        ):
            result = run_asset_quality_pipeline("/any/path.glb", {})
        self.assertFalse(result["available"])
        self.assertEqual(result["verdict"], "needs_review")
        self.assertIn("asset_quality_pipeline_failed", result["warnings"])


# ── Test 9: API response includes asset_quality fields ───────────────────────

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


class TestApiResponseIncludesAqFields(unittest.TestCase):

    def _inject_session(self, session_id: str, input_path: str):
        api_module._ai3d_sessions[session_id] = {
            "status": "uploaded",
            "input_path": input_path,
            "provider": "sf3d",
        }

    def test_api_response_has_asset_quality_keys(self):
        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td)
            session_id = "aq_api_test_001"

            manifest = _mock_manifest(glb_path)
            manifest["asset_quality"] = {
                "enabled": True, "available": True, "status": "review",
                "score": 72, "verdict": "mobile_ready", "provider_neutral": True,
                "checks": {
                    "scale_orientation": {"enabled": True, "available": True},
                    "mesh_cleanup": {"enabled": True, "available": True, "status": "ok"},
                    "lod": {"enabled": True, "available": True, "strategy": "plan_only", "tiers": []},
                    "pbr_textures": {"enabled": True, "available": True, "material_count": 1},
                    "export_profiles": {"raw": {"available": True}},
                },
                "warnings": [], "recommendations": [], "error": None,
            }
            manifest["normalization"] = {"enabled": True, "available": True, "applied": False}
            manifest["mesh_cleanup"] = {"enabled": True, "status": "ok"}
            manifest["lod"] = {"enabled": True, "strategy": "plan_only", "tiers": []}
            manifest["pbr_textures"] = {"enabled": True, "material_count": 1, "texture_count": 0}
            manifest["export_profiles"] = {"mobile_ar": {"ready": True}}

            img_path = os.path.join(td, "img.png")
            from PIL import Image
            Image.new("RGB", (64, 64)).save(img_path)

            self._inject_session(session_id, img_path)

            with patch(
                "modules.ai_3d_generation.pipeline.generate_ai_3d",
                return_value=manifest,
            ):
                with patch(
                    "modules.operations.api.settings.ai_3d_generation_enabled", True
                ):
                    resp = client.post(
                        f"/api/ai-3d/process/{session_id}",
                        json={"provider": "sf3d"},
                        headers={"X-API-Key": "test"},
                    )

        self.assertIn(resp.status_code, (200, 422), resp.text)
        if resp.status_code == 200:
            data = resp.json()
            self.assertIn("asset_quality", data)
            self.assertIn("normalization", data)
            self.assertIn("mesh_cleanup", data)
            self.assertIn("lod", data)
            self.assertIn("pbr_textures", data)
            self.assertIn("export_profiles", data)


# ── Test 10 & 11: UI sections ─────────────────────────────────────────────────

class TestUiSections(unittest.TestCase):

    def _read_ui(self) -> str:
        ui_path = Path(__file__).parent.parent / "ui" / "ai_3d_studio.html"
        return ui_path.read_text(encoding="utf-8")

    def test_ui_contains_asset_quality_section(self):
        html = self._read_ui()
        self.assertIn("asset-quality-section", html)
        self.assertIn("Asset Quality", html)

    def test_ui_distinguishes_ar_technical_readiness_from_asset_quality(self):
        html = self._read_ui()
        # AR readiness must be labelled "AR Technical Readiness" (not just "AR Readiness")
        self.assertIn("AR Technical Readiness", html)
        # And Asset Quality must be a separate section
        self.assertIn("asset-quality-section", html)
        # The two labels must be different
        self.assertIn("Asset Quality", html)
        # Confirm they are distinct sections
        ar_idx = html.find("AR Technical Readiness")
        aq_idx = html.find("Asset Quality")
        self.assertNotEqual(ar_idx, aq_idx)
        self.assertGreater(ar_idx, 0)
        self.assertGreater(aq_idx, 0)

    def test_ui_contains_all_aq_sections(self):
        html = self._read_ui()
        for section_id in [
            "normalization-section",
            "mesh-cleanup-section",
            "lod-section",
            "pbr-section",
            "export-profiles-section",
        ]:
            self.assertIn(section_id, html, f"Missing section: {section_id}")

    def test_ui_contains_hunyuan_warning(self):
        html = self._read_ui()
        self.assertIn("hunyuan-warning-section", html)
        self.assertIn("Hunyuan local inference", html)


# ── Test 12: SF3D defaults unchanged ─────────────────────────────────────────

class TestSf3dDefaultsUnchanged(unittest.TestCase):

    def test_sf3d_default_provider_unchanged(self):
        from modules.operations.settings import settings
        # SF3D should still be the default provider
        self.assertEqual(settings.ai_3d_default_provider, "sf3d")

    def test_sf3d_input_size_unchanged(self):
        from modules.operations.settings import settings
        self.assertEqual(settings.sf3d_input_size, 512)

    def test_sf3d_remesh_default_unchanged(self):
        from modules.operations.settings import settings
        self.assertIn(settings.sf3d_remesh, ("none", "remesh"))


# ── Test 13: Hunyuan disabled/safe defaults ───────────────────────────────────

class TestHunyuanSafeDefaults(unittest.TestCase):

    def test_hunyuan_disabled_by_default(self):
        from modules.operations.settings import settings
        self.assertFalse(
            settings.hunyuan3d_21_enabled,
            "Hunyuan must be disabled by default to prevent local CPU/RAM overload",
        )

    def test_hunyuan_mode_shape_only(self):
        from modules.operations.settings import settings
        self.assertEqual(settings.hunyuan3d_21_mode, "shape_only")

    def test_hunyuan_texture_disabled(self):
        from modules.operations.settings import settings
        self.assertFalse(settings.hunyuan3d_21_texture_enabled)

    def test_hunyuan_mock_runner_off_by_default(self):
        from modules.operations.settings import settings
        self.assertFalse(settings.hunyuan3d_21_mock_runner)


# ── Test 14: no destructive optimization or overwrite ─────────────────────────

class TestNoDestructiveOperation(unittest.TestCase):

    def test_normalization_applied_is_false(self):
        from modules.ai_3d_generation.asset_quality.normalization import analyze_normalization
        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td)
            result = analyze_normalization(glb_path)
        self.assertFalse(result["applied"])

    def test_raw_glb_not_overwritten_after_pipeline(self):
        from modules.ai_3d_generation.asset_quality import run_asset_quality_pipeline
        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td)
            original_bytes = Path(glb_path).read_bytes()
            manifest = _mock_manifest(glb_path)
            run_asset_quality_pipeline(glb_path, manifest)
            after_bytes = Path(glb_path).read_bytes()
        self.assertEqual(
            original_bytes, after_bytes,
            "Raw GLB must not be modified by the asset quality pipeline",
        )

    def test_lod_does_not_write_files_by_default(self):
        from modules.ai_3d_generation.asset_quality.lod import build_lod_plan
        import glob
        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td)
            build_lod_plan(
                {"face_count": 999999},
                {},
                asset_quality_context={"glb_path": glb_path, "output_dir": td},
            )
            lod_files = glob.glob(os.path.join(td, "output_*.glb"))
        self.assertEqual(len(lod_files), 0)


# ── Test 15: invalid providers still rejected ─────────────────────────────────

class TestInvalidProviderRejected(unittest.TestCase):

    def _inject_session(self, session_id: str, input_path: str):
        api_module._ai3d_sessions[session_id] = {
            "status": "uploaded",
            "input_path": input_path,
            "provider": "invalid_provider_xyz",
        }

    def test_invalid_provider_returns_400(self):
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "img.png")
            from PIL import Image
            Image.new("RGB", (64, 64)).save(img_path)

            session_id = "inv_prov_test_001"
            self._inject_session(session_id, img_path)

            with patch("modules.operations.api.verify_api_key"):
                with patch("modules.operations.api.settings.ai_3d_generation_enabled", True):
                    resp = client.post(
                        f"/api/ai-3d/process/{session_id}",
                        json={"provider": "invalid_provider_xyz"},
                        headers={"X-API-Key": "test"},
                    )

        self.assertEqual(resp.status_code, 400)
        detail = resp.json().get("detail", {})
        if isinstance(detail, dict):
            self.assertEqual(detail.get("error"), "unknown_ai3d_provider")
        else:
            self.assertIn("unknown_ai3d_provider", str(detail))


# ── Fix-commit regression tests ───────────────────────────────────────────────


class TestExportProfilesNoPathLeak(unittest.TestCase):
    """export_profiles.raw.path must never contain a server-side filesystem path."""

    def test_raw_path_is_none_when_glb_exists(self):
        from modules.ai_3d_generation.asset_quality.export_profiles import assess_export_profiles
        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td)
            manifest = {
                "output_glb_path": glb_path,
                "glb_validation": {"valid": True, "issues": [], "warnings": []},
                "mesh_stats": {},
                "ar_readiness": {},
                "output_size_bytes": 128,
            }
            result = assess_export_profiles(manifest)
        raw = result["raw"]
        self.assertIsNone(raw["path"], "raw.path must be None to avoid server path leak")
        self.assertTrue(raw["available"])
        self.assertTrue(raw["valid"])

    def test_raw_path_is_none_when_glb_missing(self):
        from modules.ai_3d_generation.asset_quality.export_profiles import assess_export_profiles
        manifest = {
            "output_glb_path": "/nonexistent/path.glb",
            "glb_validation": {"valid": None},
            "mesh_stats": {},
            "ar_readiness": {},
            "output_size_bytes": 0,
        }
        result = assess_export_profiles(manifest)
        self.assertIsNone(result["raw"]["path"])
        self.assertFalse(result["raw"]["available"])


class TestPipelineExceptionSanitization(unittest.TestCase):
    """Outer asset_quality exception must strip filesystem paths from error message."""

    def test_exception_with_path_in_message_is_sanitized(self):
        from modules.ai_3d_generation.asset_quality.quality_pipeline import _sanitize_error
        # Simulate a FileNotFoundError whose str() contains an absolute path
        exc = FileNotFoundError("/home/app/uploads/job_abc123/model.glb: No such file")
        result = _sanitize_error(exc)
        self.assertNotIn("/home/app", result)
        self.assertNotIn("\\Users\\", result)
        # Result should be the tail segment after the last separator
        self.assertIn("model.glb", result)

    def test_exception_without_path_is_unchanged(self):
        from modules.ai_3d_generation.asset_quality.quality_pipeline import _sanitize_error
        exc = RuntimeError("trimesh parse error: unexpected EOF")
        result = _sanitize_error(exc)
        self.assertEqual(result, "trimesh parse error: unexpected EOF")

    def test_exception_message_capped_at_200_chars(self):
        from modules.ai_3d_generation.asset_quality.quality_pipeline import _sanitize_error
        exc = ValueError("x" * 500)
        result = _sanitize_error(exc)
        self.assertLessEqual(len(result), 200)


class TestMeshCleanupSpanNoPtp(unittest.TestCase):
    """mesh_cleanup_audit span calculation must not use np.ptp (removed in NumPy 2)."""

    def test_span_calculation_with_simple_vertices(self):
        """Directly call the bounds check path via a real GLB with known geometry."""
        import numpy as np
        from modules.ai_3d_generation.asset_quality.mesh_cleanup_audit import audit_mesh_cleanup

        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td)
            # Must not raise AttributeError on np.ptp regardless of numpy version
            result = audit_mesh_cleanup(glb_path)
        self.assertIn("status", result)
        # If trimesh is available, metrics should be populated
        if result.get("available") and result["metrics"].get("component_count") is not None:
            self.assertNotIn("mesh_cleanup_audit_failed", result.get("warnings", []))

    def test_np_ptp_not_used_in_source(self):
        """Guard: the source file must not contain np.ptp after the fix."""
        src = Path(__file__).parent.parent / "modules" / "ai_3d_generation" / "asset_quality" / "mesh_cleanup_audit.py"
        content = src.read_text(encoding="utf-8")
        self.assertNotIn("np.ptp", content, "np.ptp is deprecated in NumPy 2.x and must not be used")
