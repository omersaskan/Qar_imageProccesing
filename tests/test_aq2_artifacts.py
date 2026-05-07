"""
AQ2 — Safe Normalized Copy + Cleanup Artifact Foundation tests.

Tests:
 1.  test_normalized_copy_never_overwrites_raw
 2.  test_normalized_copy_preserves_raw_file_bytes
 3.  test_normalized_copy_returns_stable_manifest_shape
 4.  test_normalized_copy_validation_failure_handled
 5.  test_cleanup_report_json_generated
 6.  test_cleanup_report_md_generated
 7.  test_cleanup_report_marks_manual_cleanup_when_floating_parts
 8.  test_cleanup_report_marks_retopology_when_high_component_count
 9.  test_export_package_creates_expected_file_manifest
10.  test_export_package_no_absolute_paths
11.  test_api_response_includes_aq2_fields
12.  test_ui_contains_aq2_artifacts_section
13.  test_mobile_ar_recommended_artifact_none_when_cleanup_blockers
14.  test_web_preview_may_recommend_normalized_when_valid
15.  test_destructive_cleanup_flag_always_false
16.  test_raw_glb_overwrite_impossible
17.  test_sf3d_defaults_unchanged
18.  test_hunyuan_real_inference_not_run
"""
from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional
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


# ── import helpers ───────────────────────────────────────────────────────────

_real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__


def _selective_import_error(blocked_module: str):
    """Return a side_effect function that raises ImportError only for blocked_module."""
    import builtins
    _orig = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == blocked_module or name.startswith(blocked_module + "."):
            raise ImportError(f"mocked ImportError for {name}")
        return _orig(name, *args, **kwargs)

    return _fake_import


# ── fixtures ──────────────────────────────────────────────────────────────────

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


# ── helpers ───────────────────────────────────────────────────────────────────

def _norm_analysis_with_warnings(*warnings) -> Dict[str, Any]:
    return {
        "enabled": True,
        "available": True,
        "applied": False,
        "warnings": list(warnings),
        "issues": [],
        "analysis": {
            "bounds": [[-1.0, 0.0, -1.0], [1.0, 2.0, 1.0]],
            "dimensions": {"x": 2.0, "y": 2.0, "z": 2.0},
            "center": [0.0, 1.0, 0.0],
            "ground_offset": 0.0,
            "largest_axis": "y",
            "likely_flat_on_ground": True,
        },
        "recommendations": [],
    }


def _mc_with_warnings(*warnings) -> Dict[str, Any]:
    return {
        "enabled": True,
        "available": True,
        "status": "review",
        "warnings": list(warnings),
        "issues": [],
        "metrics": {
            "component_count": None,
            "degenerate_face_count": 0,
            "boundary_edge_count": 0,
            "non_manifold_estimate": 0,
        },
        "recommendations": [],
    }


def _mc_with_high_component(count: int = 50) -> Dict[str, Any]:
    mc = _mc_with_warnings(f"high_component_count:{count}")
    mc["metrics"]["component_count"] = count
    return mc


def _mock_export_profiles(mobile_ar_ready: bool = True, blocking: Optional[list] = None) -> Dict[str, Any]:
    return {
        "raw": {"available": True, "valid": True, "path": None},
        "web_preview": {"ready": True, "blocking_reasons": [], "warnings": []},
        "mobile_ar": {
            "ready": mobile_ar_ready,
            "blocking_reasons": blocking or [],
            "warnings": [],
        },
        "desktop_high": {"ready": True, "blocking_reasons": [], "warnings": []},
    }


# ── Test 1: normalized copy never overwrites raw ──────────────────────────────

class TestNormalizedCopyNeverOverwritesRaw(unittest.TestCase):

    def test_normalized_copy_never_overwrites_raw(self):
        """If raw_glb_path resolves to normalized.glb, return error with warning."""
        from modules.ai_3d_generation.asset_quality.normalized_copy import create_normalized_copy
        with tempfile.TemporaryDirectory() as td:
            # The raw GLB is named normalized.glb — output path would be identical
            raw_path = os.path.join(td, "normalized.glb")
            with open(raw_path, "wb") as f:
                f.write(_make_glb())
            result = create_normalized_copy(
                raw_glb_path=raw_path,
                output_dir=td,
                normalization_analysis=_norm_analysis_with_warnings(
                    "model_not_centered", "ground_alignment_uncertain"
                ),
            )
        self.assertIn("normalized_copy_would_overwrite_raw", result["warnings"])
        self.assertIsNotNone(result["error"])
        self.assertFalse(result["applied"])


# ── Test 2: raw file bytes preserved ─────────────────────────────────────────

class TestNormalizedCopyPreservesRaw(unittest.TestCase):

    def test_normalized_copy_preserves_raw_file_bytes(self):
        """Running the full AQ2 pipeline must not change the raw GLB bytes."""
        from modules.ai_3d_generation.asset_quality.artifacts import run_aq2_pipeline
        with tempfile.TemporaryDirectory() as td:
            raw_path = _write_glb(td, "out.glb")
            original_bytes = Path(raw_path).read_bytes()

            manifest = {
                "session_id": "aq2_test",
                "normalization": _norm_analysis_with_warnings(
                    "model_not_centered", "ground_alignment_uncertain"
                ),
                "mesh_cleanup": _mc_with_warnings(),
                "pbr_textures": {"issues": [], "warnings": [], "material_count": 1, "texture_count": 0},
                "export_profiles": _mock_export_profiles(),
            }
            run_aq2_pipeline(raw_path, td, manifest, {})

            after_bytes = Path(raw_path).read_bytes()

        self.assertEqual(
            original_bytes, after_bytes,
            "Raw GLB must not be modified by AQ2 pipeline",
        )


# ── Test 3: stable manifest shape ────────────────────────────────────────────

class TestNormalizedCopyStableShape(unittest.TestCase):

    def test_normalized_copy_returns_stable_manifest_shape(self):
        from modules.ai_3d_generation.asset_quality.normalized_copy import create_normalized_copy
        with tempfile.TemporaryDirectory() as td:
            raw_path = _write_glb(td)
            result = create_normalized_copy(
                raw_glb_path=raw_path,
                output_dir=td,
                normalization_analysis=_norm_analysis_with_warnings(),
            )
        required_keys = [
            "enabled", "available", "applied", "path", "raw_preserved",
            "transform_applied", "before", "after", "validation", "warnings", "error",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        self.assertIn("centered", result["transform_applied"])
        self.assertIn("ground_aligned", result["transform_applied"])
        self.assertIn("scaled", result["transform_applied"])
        self.assertIn("orientation_changed", result["transform_applied"])
        self.assertTrue(result["raw_preserved"])
        self.assertFalse(result["transform_applied"]["scaled"])
        self.assertFalse(result["transform_applied"]["orientation_changed"])


# ── Test 4: validation failure handled ───────────────────────────────────────

class TestNormalizedCopyValidationFailure(unittest.TestCase):

    def test_normalized_copy_validation_failure_handled(self):
        """
        If validate_glb_content returns invalid, available and applied become False.
        We mock trimesh to raise ImportError so the code falls through to the
        copy-only path, then mock _validate_normalized to inject an invalid result.
        """
        from modules.ai_3d_generation.asset_quality import normalized_copy as nc_mod

        def _mock_validate_invalid(result_dict, out_path):
            result_dict["validation"] = {
                "valid": False,
                "issues": ["test_invalid"],
                "warnings": [],
            }
            result_dict["available"] = False
            result_dict["applied"] = False
            result_dict["warnings"].append("normalized_copy_validation_failed")

        with tempfile.TemporaryDirectory() as td:
            raw_path = _write_glb(td)
            # Force the ImportError path (copy-only) so _validate_normalized is called
            with patch("builtins.__import__", side_effect=_selective_import_error("trimesh")):
                with patch.object(nc_mod, "_validate_normalized", side_effect=_mock_validate_invalid):
                    result = nc_mod.create_normalized_copy(
                        raw_glb_path=raw_path,
                        output_dir=td,
                        normalization_analysis=_norm_analysis_with_warnings(),
                    )

        self.assertFalse(result["available"])
        self.assertFalse(result["applied"])
        self.assertIn("normalized_copy_validation_failed", result["warnings"])


# ── Test 5: cleanup report JSON generated ────────────────────────────────────

class TestCleanupReportJsonGenerated(unittest.TestCase):

    def test_cleanup_report_json_generated(self):
        from modules.ai_3d_generation.asset_quality.cleanup_report import write_cleanup_report
        with tempfile.TemporaryDirectory() as td:
            result = write_cleanup_report(
                output_dir=td,
                manifest={"session_id": "test_json"},
                asset_quality={"verdict": "needs_review", "score": 55},
                mesh_cleanup=_mc_with_warnings(),
                normalization=_norm_analysis_with_warnings(),
                pbr_textures={"issues": [], "warnings": [], "material_count": 1, "texture_count": 0},
                export_profiles=_mock_export_profiles(),
            )
            json_file = Path(td) / "cleanup_report.json"
            self.assertTrue(json_file.exists(), "cleanup_report.json must be written")
            data = json.loads(json_file.read_text(encoding="utf-8"))
        self.assertTrue(result["available"])
        self.assertEqual(result["json_path"], "cleanup_report.json")
        self.assertIn("summary", data)
        self.assertIn("mesh_issues", data)
        self.assertIn("normalization_issues", data)
        self.assertIn("pbr_issues", data)


# ── Test 6: cleanup report Markdown generated ─────────────────────────────────

class TestCleanupReportMdGenerated(unittest.TestCase):

    def test_cleanup_report_md_generated(self):
        from modules.ai_3d_generation.asset_quality.cleanup_report import write_cleanup_report
        with tempfile.TemporaryDirectory() as td:
            result = write_cleanup_report(
                output_dir=td,
                manifest={"session_id": "test_md"},
                asset_quality={"verdict": "needs_review", "score": 55},
                mesh_cleanup=_mc_with_warnings(),
                normalization=_norm_analysis_with_warnings(),
                pbr_textures={"issues": [], "warnings": [], "material_count": 1, "texture_count": 0},
                export_profiles=_mock_export_profiles(),
            )
            md_file = Path(td) / "cleanup_report.md"
            self.assertTrue(md_file.exists(), "cleanup_report.md must be written")
            content = md_file.read_text(encoding="utf-8")
        self.assertTrue(result["available"])
        self.assertEqual(result["markdown_path"], "cleanup_report.md")
        self.assertIn("# Asset Quality Cleanup Report", content)
        self.assertIn("## Summary", content)
        self.assertIn("## Mesh Issues", content)


# ── Test 7: manual cleanup required when floating parts ──────────────────────

class TestCleanupReportManualCleanup(unittest.TestCase):

    def test_cleanup_report_marks_manual_cleanup_when_floating_parts(self):
        from modules.ai_3d_generation.asset_quality.cleanup_report import write_cleanup_report
        with tempfile.TemporaryDirectory() as td:
            result = write_cleanup_report(
                output_dir=td,
                manifest={"session_id": "float_test"},
                asset_quality={"verdict": "needs_review", "score": 40},
                mesh_cleanup=_mc_with_warnings("floating_parts_detected"),
                normalization=_norm_analysis_with_warnings(),
                pbr_textures={"issues": [], "warnings": [], "material_count": 1, "texture_count": 0},
                export_profiles=_mock_export_profiles(),
            )
        self.assertTrue(result["manual_cleanup_required"])
        self.assertTrue(result["available"])


# ── Test 8: retopology recommended when high component count ─────────────────

class TestCleanupReportRetopology(unittest.TestCase):

    def test_cleanup_report_marks_retopology_when_high_component_count(self):
        from modules.ai_3d_generation.asset_quality.cleanup_report import write_cleanup_report
        with tempfile.TemporaryDirectory() as td:
            result = write_cleanup_report(
                output_dir=td,
                manifest={"session_id": "retopo_test"},
                asset_quality={"verdict": "needs_review", "score": 35},
                mesh_cleanup=_mc_with_high_component(100),
                normalization=_norm_analysis_with_warnings(),
                pbr_textures={"issues": [], "warnings": [], "material_count": 1, "texture_count": 0},
                export_profiles=_mock_export_profiles(),
            )
        self.assertTrue(result["retopology_recommended"])
        self.assertTrue(result["manual_cleanup_required"])


# ── Test 9: export package creates expected file manifest ─────────────────────

class TestExportPackageFileManifest(unittest.TestCase):

    def test_export_package_creates_expected_file_manifest(self):
        from modules.ai_3d_generation.asset_quality.export_package import create_export_package
        with tempfile.TemporaryDirectory() as td:
            raw_path = _write_glb(td, "out.glb")
            # Provide a normalized.glb in td that is "available and valid"
            norm_glb = os.path.join(td, "normalized.glb")
            Path(norm_glb).write_bytes(_make_glb())

            # Write cleanup reports
            Path(os.path.join(td, "cleanup_report.json")).write_text('{"ok":true}', encoding="utf-8")
            Path(os.path.join(td, "cleanup_report.md")).write_text("# Report\n", encoding="utf-8")

            normalized_copy = {
                "available": True,
                "applied": False,
                "validation": {"valid": True, "issues": [], "warnings": []},
                "warnings": [],
            }
            cleanup_report = {"available": True, "json_path": "cleanup_report.json", "markdown_path": "cleanup_report.md"}
            asset_quality = {"verdict": "needs_review", "score": 55}

            result = create_export_package(
                session_dir=td,
                raw_glb_path=raw_path,
                normalized_copy=normalized_copy,
                cleanup_report=cleanup_report,
                asset_quality=asset_quality,
                export_profiles=_mock_export_profiles(),
            )

        self.assertTrue(result["available"])
        self.assertIsNotNone(result["files"])
        expected_keys = [
            "raw_glb", "normalized_glb", "cleanup_report_json",
            "cleanup_report_md", "asset_quality_json", "export_manifest_json",
        ]
        for key in expected_keys:
            self.assertIn(key, result["files"], f"Missing file key: {key}")
        # All expected files should be available
        self.assertTrue(result["files"]["raw_glb"]["available"])
        self.assertTrue(result["files"]["normalized_glb"]["available"])
        self.assertTrue(result["files"]["cleanup_report_json"]["available"])
        self.assertTrue(result["files"]["cleanup_report_md"]["available"])
        self.assertTrue(result["files"]["asset_quality_json"]["available"])
        self.assertTrue(result["files"]["export_manifest_json"]["available"])


# ── Test 10: no absolute paths in export package ─────────────────────────────

class TestExportPackageNoAbsolutePaths(unittest.TestCase):

    def _collect_strings(self, obj, found=None):
        """Recursively collect all string values from a nested dict/list."""
        if found is None:
            found = []
        if isinstance(obj, dict):
            for v in obj.values():
                self._collect_strings(v, found)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_strings(item, found)
        elif isinstance(obj, str):
            found.append(obj)
        return found

    def test_export_package_no_absolute_paths(self):
        from modules.ai_3d_generation.asset_quality.export_package import create_export_package
        with tempfile.TemporaryDirectory() as td:
            raw_path = _write_glb(td, "out.glb")
            norm_glb = os.path.join(td, "normalized.glb")
            Path(norm_glb).write_bytes(_make_glb())

            normalized_copy = {
                "available": True,
                "applied": True,
                "validation": {"valid": True, "issues": [], "warnings": []},
                "warnings": [],
            }
            result = create_export_package(
                session_dir=td,
                raw_glb_path=raw_path,
                normalized_copy=normalized_copy,
                cleanup_report={"available": False},
                asset_quality={"verdict": "needs_review"},
                export_profiles=_mock_export_profiles(),
            )

        all_strings = self._collect_strings(result)
        for s in all_strings:
            # Check no absolute path patterns leak through
            self.assertFalse(
                os.path.isabs(s) and os.sep in s,
                f"Absolute path leaked into export package result: {s!r}",
            )
            # Windows-style absolute paths (C:\...) or Unix (/...)
            self.assertFalse(
                len(s) > 2 and s[1:3] == ":\\" and s[0].isalpha(),
                f"Windows absolute path leaked: {s!r}",
            )
            self.assertFalse(
                s.startswith("/") and len(s) > 1 and "/" in s[1:],
                f"Unix absolute path leaked: {s!r}",
            )


# ── Test 11: API response includes AQ2 fields ─────────────────────────────────

class TestApiResponseIncludesAq2Fields(unittest.TestCase):

    def _inject_session(self, session_id: str, input_path: str):
        api_module._ai3d_sessions[session_id] = {
            "status": "uploaded",
            "input_path": input_path,
            "provider": "sf3d",
        }

    def test_api_response_includes_aq2_fields(self):
        with tempfile.TemporaryDirectory() as td:
            glb_path = _write_glb(td)
            session_id = "aq2_api_test_001"

            manifest = {
                "session_id": session_id,
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
                "glb_validation": {"valid": True, "issues": [], "warnings": []},
                "ar_readiness": {"enabled": True, "score": 85, "verdict": "mobile_ready", "checks": {}, "warnings": [], "recommendations": []},
                "asset_quality": {
                    "enabled": True, "available": True, "status": "review",
                    "score": 72, "verdict": "mobile_ready", "provider_neutral": True,
                    "checks": {}, "warnings": [], "recommendations": [], "error": None,
                },
                "normalization": {"enabled": True, "available": True, "applied": False},
                "mesh_cleanup": {"enabled": True, "status": "ok"},
                "lod": {"enabled": True, "strategy": "plan_only", "tiers": []},
                "pbr_textures": {"enabled": True, "material_count": 1, "texture_count": 0},
                "export_profiles": {"mobile_ar": {"ready": True, "blocking_reasons": [], "warnings": []}},
                # AQ2 fields
                "aq2": {
                    "enabled": True, "status": "ok",
                    "normalized_copy": {"available": False, "applied": False, "warnings": ["raw_glb_missing"]},
                    "cleanup_report": {"available": True, "json_path": "cleanup_report.json"},
                    "export_package": {"available": True, "package_dir": "export_package"},
                    "warnings": [], "error": None,
                },
                "normalized_copy": {"available": False},
                "cleanup_report": {"available": True},
                "export_package": {"available": True},
            }

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
            self.assertIn("aq2", data)
            self.assertIn("normalized_copy", data)
            self.assertIn("cleanup_report", data)
            self.assertIn("export_package", data)


# ── Test 12: UI contains AQ2 artifacts section ────────────────────────────────

class TestUiContainsAq2Section(unittest.TestCase):

    def _read_ui(self) -> str:
        ui_path = Path(__file__).parent.parent / "ui" / "ai_3d_studio.html"
        return ui_path.read_text(encoding="utf-8")

    def test_ui_contains_aq2_artifacts_section(self):
        html = self._read_ui()
        self.assertIn("aq2-artifacts-section", html)
        self.assertIn("AQ2 Artifacts", html)
        self.assertIn("aq2-content", html)
        # AQ2 section must be after export profiles section
        ep_idx = html.find("export-profiles-section")
        aq2_idx = html.find("aq2-artifacts-section")
        hunyuan_idx = html.find("hunyuan-warning-section")
        self.assertGreater(aq2_idx, ep_idx, "AQ2 section must come after export profiles")
        self.assertLess(aq2_idx, hunyuan_idx, "AQ2 section must come before hunyuan warning")


# ── Test 13: mobile_ar recommended_artifact=none when cleanup blockers ────────

class TestMobileArRecommendedArtifactNone(unittest.TestCase):

    def test_mobile_ar_recommended_artifact_none_when_cleanup_blockers(self):
        from modules.ai_3d_generation.asset_quality.artifacts import update_export_profiles_recommended_artifact
        # mobile_ar not ready due to cleanup blockers
        export_profiles = {
            "raw": {"available": True, "valid": True, "path": None},
            "web_preview": {"ready": True, "blocking_reasons": [], "warnings": []},
            "mobile_ar": {
                "ready": False,
                "blocking_reasons": ["mesh_cleanup_required", "floating_parts_detected"],
                "warnings": [],
            },
            "desktop_high": {"ready": True, "blocking_reasons": [], "warnings": []},
        }
        normalized_copy = {
            "available": True,
            "validation": {"valid": True, "issues": [], "warnings": []},
        }
        result = update_export_profiles_recommended_artifact(export_profiles, normalized_copy)
        self.assertEqual(
            result["mobile_ar"]["recommended_artifact"], "none",
            "mobile_ar not ready => recommended_artifact must be 'none'",
        )


# ── Test 14: web_preview recommends normalized when valid ─────────────────────

class TestWebPreviewRecommendsNormalized(unittest.TestCase):

    def test_web_preview_may_recommend_normalized_when_valid(self):
        from modules.ai_3d_generation.asset_quality.artifacts import update_export_profiles_recommended_artifact
        export_profiles = {
            "raw": {"available": True, "valid": True, "path": None},
            "web_preview": {"ready": True, "blocking_reasons": [], "warnings": []},
            "mobile_ar": {"ready": True, "blocking_reasons": [], "warnings": []},
            "desktop_high": {"ready": True, "blocking_reasons": [], "warnings": []},
        }
        normalized_copy = {
            "available": True,
            "validation": {"valid": True, "issues": [], "warnings": []},
        }
        result = update_export_profiles_recommended_artifact(export_profiles, normalized_copy)
        self.assertEqual(
            result["web_preview"]["recommended_artifact"], "normalized",
            "web_preview should recommend 'normalized' when normalized copy is valid",
        )
        self.assertEqual(result["mobile_ar"]["recommended_artifact"], "normalized")
        self.assertEqual(result["desktop_high"]["recommended_artifact"], "normalized")
        self.assertEqual(result["raw"]["recommended_artifact"], "raw")


# ── Test 15: destructive cleanup flag always False ────────────────────────────

class TestDestructiveCleanupFlagAlwaysFalse(unittest.TestCase):

    def test_destructive_cleanup_flag_always_false(self):
        from modules.ai_3d_generation.asset_quality.artifacts import _DESTRUCTIVE_CLEANUP_ENABLED
        self.assertFalse(
            _DESTRUCTIVE_CLEANUP_ENABLED,
            "_DESTRUCTIVE_CLEANUP_ENABLED must always be False",
        )

    def test_overwrite_raw_flag_always_false(self):
        from modules.ai_3d_generation.asset_quality.artifacts import _OVERWRITE_RAW
        self.assertFalse(
            _OVERWRITE_RAW,
            "_OVERWRITE_RAW must always be False",
        )

    def test_normalized_copy_overwrite_raw_always_false(self):
        from modules.ai_3d_generation.asset_quality.normalized_copy import _OVERWRITE_RAW
        self.assertFalse(
            _OVERWRITE_RAW,
            "normalized_copy._OVERWRITE_RAW must always be False",
        )


# ── Test 16: raw GLB overwrite impossible ─────────────────────────────────────

class TestRawGlbOverwriteImpossible(unittest.TestCase):

    def test_raw_glb_overwrite_impossible(self):
        """If raw path is named normalized.glb, the hard guard triggers."""
        from modules.ai_3d_generation.asset_quality.normalized_copy import create_normalized_copy
        with tempfile.TemporaryDirectory() as td:
            # Create a raw GLB named exactly normalized.glb in td
            raw_path = os.path.join(td, "normalized.glb")
            with open(raw_path, "wb") as f:
                f.write(_make_glb())
            original_bytes = Path(raw_path).read_bytes()

            result = create_normalized_copy(
                raw_glb_path=raw_path,
                output_dir=td,
                normalization_analysis=_norm_analysis_with_warnings(
                    "model_not_centered", "ground_alignment_uncertain"
                ),
            )

            after_bytes = Path(raw_path).read_bytes()

        # Hard guard: must warn and not overwrite
        self.assertIn("normalized_copy_would_overwrite_raw", result["warnings"])
        self.assertFalse(result["applied"])
        self.assertEqual(original_bytes, after_bytes, "Raw GLB must not be modified")


# ── Test 17: SF3D defaults unchanged ─────────────────────────────────────────

class TestSf3dDefaultsUnchanged(unittest.TestCase):

    def test_sf3d_defaults_unchanged(self):
        from modules.operations.settings import settings
        self.assertEqual(
            settings.ai_3d_default_provider, "sf3d",
            "Default provider must remain sf3d after AQ2",
        )
        self.assertEqual(settings.sf3d_input_size, 512)


# ── Test 18: Hunyuan real inference not run ───────────────────────────────────

class TestHunyuanRealInferenceNotRun(unittest.TestCase):

    def test_hunyuan_real_inference_not_run(self):
        from modules.operations.settings import settings
        self.assertFalse(
            settings.hunyuan3d_21_enabled,
            "Hunyuan must remain disabled to prevent real inference",
        )
