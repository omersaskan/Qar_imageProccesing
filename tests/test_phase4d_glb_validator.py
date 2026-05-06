"""
Phase 4D tests — pure-Python GLB structural validator and its integration
with postprocessing, AR readiness, and the API process response.
"""
import json
import struct
import tempfile
import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from modules.operations.api import app
import modules.operations.api as api_module

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_glb(gltf_dict=None, version=2, corrupt_magic=False,
              truncate_at=None, length_override=None) -> bytes:
    """Build a minimal GLB binary for testing."""
    if gltf_dict is None:
        gltf_dict = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        }
    json_bytes = json.dumps(gltf_dict).encode("utf-8")
    pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * pad

    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    total_length = 12 + len(json_chunk)
    magic = 0x00000000 if corrupt_magic else 0x46546C67
    if length_override is not None:
        total_length = length_override
    header = struct.pack("<III", magic, version, total_length)
    data = header + json_chunk
    if truncate_at is not None:
        data = data[:truncate_at]
    return data


def _write_glb(tmp_dir, name="out.glb", **kwargs) -> str:
    path = os.path.join(tmp_dir, name)
    with open(path, "wb") as f:
        f.write(_make_glb(**kwargs))
    return path


# ── validate_glb_content ──────────────────────────────────────────────────────

class TestValidateGlbContent(unittest.TestCase):

    def _validate(self, path):
        from modules.qa_validation.gltf_validator import validate_glb_content
        return validate_glb_content(path)

    def test_valid_minimal_glb_passes(self):
        with tempfile.TemporaryDirectory() as td:
            path = _write_glb(td)
            r = self._validate(path)
        self.assertTrue(r["valid"])
        self.assertFalse(r["issues"])
        self.assertEqual(r["error"], None)
        self.assertTrue(r["available"])
        self.assertEqual(r["metadata"]["mesh_count"], 1)
        self.assertEqual(r["metadata"]["scene_count"], 1)

    def test_missing_file_returns_glb_missing(self):
        r = self._validate("/no/such/file.glb")
        self.assertFalse(r["valid"])
        self.assertIn("glb_missing", r["issues"])
        self.assertEqual(r["error"], "glb_missing")

    def test_none_path_returns_glb_missing(self):
        r = self._validate(None)
        self.assertFalse(r["valid"])
        self.assertIn("glb_missing", r["issues"])

    def test_corrupt_magic_fails(self):
        with tempfile.TemporaryDirectory() as td:
            path = _write_glb(td, corrupt_magic=True)
            r = self._validate(path)
        self.assertFalse(r["valid"])
        self.assertIn("invalid_magic", r["issues"])
        self.assertEqual(r["error"], "invalid_magic")

    def test_truncated_file_fails(self):
        with tempfile.TemporaryDirectory() as td:
            path = _write_glb(td, truncate_at=8)  # truncate mid-header
            r = self._validate(path)
        self.assertFalse(r["valid"])
        self.assertIn("file_too_small_for_header", r["issues"])

    def test_glb_with_no_meshes_fails(self):
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{}],
            # no "meshes" key
        }
        with tempfile.TemporaryDirectory() as td:
            path = _write_glb(td, gltf_dict=gltf)
            r = self._validate(path)
        self.assertFalse(r["valid"])
        self.assertIn("missing_meshes", r["issues"])

    def test_glb_with_no_scenes_fails(self):
        gltf = {
            "asset": {"version": "2.0"},
            "meshes": [{"primitives": []}],
            # no "scenes"
        }
        with tempfile.TemporaryDirectory() as td:
            path = _write_glb(td, gltf_dict=gltf)
            r = self._validate(path)
        self.assertFalse(r["valid"])
        self.assertIn("no_scenes", r["issues"])

    def test_length_mismatch_adds_issue(self):
        with tempfile.TemporaryDirectory() as td:
            path = _write_glb(td, length_override=9999)
            r = self._validate(path)
        self.assertIn("length_mismatch", r["issues"])

    def test_result_has_required_keys(self):
        with tempfile.TemporaryDirectory() as td:
            path = _write_glb(td)
            r = self._validate(path)
        for k in ("enabled", "available", "valid", "issues", "warnings", "metadata", "error"):
            self.assertIn(k, r)

    def test_metadata_counts_populated(self):
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0, 1]}],
            "nodes": [{"mesh": 0}, {}],
            "meshes": [{"primitives": []}, {"primitives": []}],
            "materials": [{}],
            "textures": [{}],
            "images": [{}],
        }
        with tempfile.TemporaryDirectory() as td:
            path = _write_glb(td, gltf_dict=gltf)
            r = self._validate(path)
        self.assertEqual(r["metadata"]["mesh_count"], 2)
        self.assertEqual(r["metadata"]["material_count"], 1)
        self.assertEqual(r["metadata"]["texture_count"], 1)
        self.assertEqual(r["metadata"]["image_count"], 1)
        self.assertEqual(r["metadata"]["node_count"], 2)

    def test_invalid_buffer_view_ref_fails(self):
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": []}],
            "buffers": [{"byteLength": 100}],
            "bufferViews": [{"buffer": 99, "byteLength": 10}],  # buffer index 99 out of range
        }
        with tempfile.TemporaryDirectory() as td:
            path = _write_glb(td, gltf_dict=gltf)
            r = self._validate(path)
        self.assertFalse(r["valid"])
        self.assertTrue(any("buffer_view_" in i for i in r["issues"]))

    def test_not_a_real_glb_file_raw_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.glb")
            with open(path, "wb") as f:
                f.write(b"This is not a GLB file at all!")
            r = self._validate(path)
        self.assertFalse(r["valid"])
        self.assertIn("invalid_magic", r["issues"])


# ── postprocess integration ───────────────────────────────────────────────────

class TestPostprocessValidateIntegration(unittest.TestCase):

    def test_validate_glb_if_available_valid_glb(self):
        from modules.ai_3d_generation.postprocess import validate_glb_if_available
        with tempfile.TemporaryDirectory() as td:
            path = _write_glb(td)
            result = validate_glb_if_available(path)
        self.assertTrue(result["applied"])
        self.assertIn("result", result)
        self.assertTrue(result["result"]["valid"])

    def test_validate_glb_if_available_invalid_glb(self):
        from modules.ai_3d_generation.postprocess import validate_glb_if_available
        with tempfile.TemporaryDirectory() as td:
            path = _write_glb(td, corrupt_magic=True)
            result = validate_glb_if_available(path)
        self.assertTrue(result["applied"])
        self.assertFalse(result["result"]["valid"])

    def test_validate_glb_if_available_missing_file(self):
        from modules.ai_3d_generation.postprocess import validate_glb_if_available
        result = validate_glb_if_available("/nonexistent/path/out.glb")
        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "glb_missing")

    def test_run_postprocess_validate_result_present(self):
        from modules.ai_3d_generation.postprocess import run_postprocess
        with tempfile.TemporaryDirectory() as td:
            path = _write_glb(td)
            result = run_postprocess(path, enabled=True)
        self.assertTrue(result["enabled"])
        self.assertTrue(result["validate"]["applied"])
        self.assertIn("result", result["validate"])
        self.assertTrue(result["validate"]["result"]["valid"])


# ── AR readiness: invalid GLB blocks mobile_ready ─────────────────────────────

class TestArReadinessGlbValidation(unittest.TestCase):

    def _base_manifest(self, glb_path):
        return {
            "output_glb_path": glb_path,
            "output_size_bytes": 1024,
            "provider_status": "ok",
            "quality_gate": {"verdict": "ok"},
            "worker_metadata": {"texture_resolution": 1024},
            "review_required": False,
        }

    def test_invalid_glb_blocks_mobile_ready(self):
        import tempfile, os
        from modules.ai_3d_generation.ar_readiness import assess_ar_readiness
        with tempfile.TemporaryDirectory() as td:
            glb = os.path.join(td, "out.glb")
            open(glb, "wb").write(_make_glb())
            m = self._base_manifest(glb)
            m["glb_validation"] = {"valid": False, "issues": ["missing_meshes"], "warnings": []}
            r = assess_ar_readiness(m)
        self.assertNotEqual(r["verdict"], "mobile_ready")
        self.assertIn("glb_validation_failed", r["warnings"])
        self.assertFalse(r["checks"]["glb_validation"]["ok"])

    def test_valid_glb_does_not_penalise(self):
        import tempfile, os
        from modules.ai_3d_generation.ar_readiness import assess_ar_readiness
        with tempfile.TemporaryDirectory() as td:
            glb = os.path.join(td, "out.glb")
            open(glb, "wb").write(_make_glb())
            m = self._base_manifest(glb)
            m["glb_validation"] = {"valid": True, "issues": [], "warnings": []}
            r = assess_ar_readiness(m)
        self.assertNotIn("glb_validation_failed", r["warnings"])
        self.assertTrue(r["checks"]["glb_validation"]["ok"])
        self.assertEqual(r["verdict"], "mobile_ready")

    def test_absent_glb_validation_does_not_penalise(self):
        import tempfile, os
        from modules.ai_3d_generation.ar_readiness import assess_ar_readiness
        with tempfile.TemporaryDirectory() as td:
            glb = os.path.join(td, "out.glb")
            open(glb, "wb").write(_make_glb())
            m = self._base_manifest(glb)
            # no glb_validation key — must not penalise
            r = assess_ar_readiness(m)
        self.assertNotIn("glb_validation_failed", r["warnings"])
        self.assertIsNone(r["checks"]["glb_validation"]["ok"])

    def test_invalid_glb_lowers_score(self):
        import tempfile, os
        from modules.ai_3d_generation.ar_readiness import assess_ar_readiness
        with tempfile.TemporaryDirectory() as td:
            glb = os.path.join(td, "out.glb")
            open(glb, "wb").write(_make_glb())
            m = self._base_manifest(glb)
            m["glb_validation"] = {"valid": False, "issues": ["missing_meshes"], "warnings": []}
            r = assess_ar_readiness(m)
        self.assertLess(r["score"], 80)


# ── API response includes glb_validation ─────────────────────────────────────

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


def _inject_session(session_id, input_path):
    api_module._ai3d_sessions[session_id] = {
        "status": "uploaded",
        "input_path": input_path,
        "provider": "sf3d",
    }


def test_api_process_response_includes_glb_validation(tmp_path):
    sid = "test_4d_val"
    inp = str(tmp_path / "upload.png")
    Path(inp).touch()
    _inject_session(sid, inp)

    manifest = {
        "session_id": sid, "status": "review",
        "provider": "sf3d", "provider_status": "ok",
        "execution_mode": "wsl_subprocess",
        "output_glb_path": str(tmp_path / "output.glb"),
        "output_size_bytes": 1024, "peak_mem_mb": 100.0,
        "worker_metadata": {}, "input_type": "single_image",
        "input_mode": "single_image", "candidate_count": 1,
        "selected_candidate_id": "c1", "candidate_ranking": [],
        "warnings": [], "errors": [], "review_required": False,
        "is_true_scan": False, "geometry_confidence": "estimated",
        "model_name": "sf3d", "quality_mode": "high",
        "resolved_quality": {}, "preprocessing": {}, "postprocessing": {},
        "quality_gate": {"verdict": "review"},
        "glb_validation": {"valid": True, "issues": [], "warnings": []},
    }

    with patch("modules.ai_3d_generation.pipeline.generate_ai_3d", return_value=manifest), \
         patch("modules.ai_3d_generation.multi_input.load_session_inputs", return_value=None):
        r = client.post(f"/api/ai-3d/process/{sid}", json={"options": {}})

    assert r.status_code == 200
    data = r.json()
    assert "glb_validation" in data
    assert data["glb_validation"]["valid"] is True
    assert data["glb_validation"]["issues"] == []


def test_api_process_glb_validation_none_when_absent(tmp_path):
    sid = "test_4d_absent"
    inp = str(tmp_path / "upload.png")
    Path(inp).touch()
    _inject_session(sid, inp)

    manifest = {
        "session_id": sid, "status": "review",
        "provider": "sf3d", "provider_status": "ok",
        "execution_mode": "wsl_subprocess",
        "output_glb_path": None, "output_size_bytes": 0,
        "peak_mem_mb": 0, "worker_metadata": {},
        "input_type": "single_image", "input_mode": "single_image",
        "candidate_count": 0, "selected_candidate_id": None,
        "candidate_ranking": [], "warnings": [], "errors": [],
        "review_required": True, "is_true_scan": False,
        "geometry_confidence": "estimated", "model_name": "sf3d",
        "quality_mode": "high", "resolved_quality": {},
        "preprocessing": {}, "postprocessing": {},
        "quality_gate": {"verdict": "failed"},
        # no glb_validation key
    }

    with patch("modules.ai_3d_generation.pipeline.generate_ai_3d", return_value=manifest), \
         patch("modules.ai_3d_generation.multi_input.load_session_inputs", return_value=None):
        r = client.post(f"/api/ai-3d/process/{sid}", json={"options": {}})

    assert r.status_code == 200
    data = r.json()
    assert "glb_validation" in data
    assert data["glb_validation"] is None


# ── Guard: no optimization, no external providers ─────────────────────────────

class TestPhase4DGuards(unittest.TestCase):

    def test_no_glb_optimization_in_postprocess(self):
        from modules.ai_3d_generation.postprocess import run_postprocess
        with tempfile.TemporaryDirectory() as td:
            path = _write_glb(td)
            result = run_postprocess(path, enabled=True)
        self.assertFalse(result["optimize"]["applied"])

    def test_no_external_provider_defaults(self):
        from modules.operations.settings import Settings
        s = Settings(_env_file=None)
        self.assertFalse(getattr(s, "meshy_enabled", False))
        self.assertFalse(getattr(s, "rodin_enabled", False))
        self.assertFalse(getattr(s, "ai_3d_remote_providers_enabled", False))

    def test_validate_glb_content_does_not_write_files(self):
        from modules.qa_validation.gltf_validator import validate_glb_content
        with tempfile.TemporaryDirectory() as td:
            path = _write_glb(td)
            before = set(os.listdir(td))
            validate_glb_content(path)
            after = set(os.listdir(td))
        self.assertEqual(before, after, "validate_glb_content must not create files")


if __name__ == "__main__":
    unittest.main()
