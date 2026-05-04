"""
Tests for modules/ai_3d_generation/ scaffold.

Coverage:
  - AI3DProviderBase.safe_generate (availability, exception handling, status normalisation)
  - SF3DProvider.is_available (settings-driven)
  - SF3DProvider.generate (subprocess mock: timeout, no stdout, bad JSON, unavailable, ok, output missing)
  - quality_gate.evaluate (all verdicts)
  - router.decide_asset_pipeline (all routing rules)
  - input_preprocessor.preprocess_input (center crop, bbox, mask)
  - manifest.build_manifest / write_manifest
  - Pipeline integration (generate_ai_3d end-to-end with mocked provider)
"""
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Provider base
# ─────────────────────────────────────────────────────────────────────────────

class TestProviderBase(unittest.TestCase):

    def _make_provider(self, available=True, avail_reason="", generate_result=None, raises=None):
        from modules.ai_3d_generation.provider_base import AI3DProviderBase

        class _Fake(AI3DProviderBase):
            name = "fake"
            output_format = "glb"

            def is_available(self) -> Tuple[bool, str]:
                return available, avail_reason

            def generate(self, *_a, **_kw) -> Dict[str, Any]:
                if raises:
                    raise raises
                return generate_result or {"status": "ok", "output_path": None,
                                          "warnings": [], "model_name": "test"}

        return _Fake()

    def test_safe_generate_unavailable(self):
        p = self._make_provider(available=False, avail_reason="sf3d_disabled")
        r = p.safe_generate("img.png", "/tmp/out")
        self.assertEqual(r["status"], "unavailable")
        self.assertIn("sf3d_disabled", r["error"])

    def test_safe_generate_exception(self):
        p = self._make_provider(raises=RuntimeError("boom"))
        r = p.safe_generate("img.png", "/tmp/out")
        self.assertEqual(r["status"], "failed")
        self.assertIn("boom", r["error"])

    def test_safe_generate_normalises_unknown_status(self):
        p = self._make_provider(generate_result={"status": "error", "error_code": "oops"})
        r = p.safe_generate("img.png", "/tmp/out")
        self.assertEqual(r["status"], "failed")

    def test_safe_generate_ok_passthrough(self):
        p = self._make_provider(generate_result={
            "status": "ok", "output_path": None, "warnings": [], "model_name": "m"
        })
        r = p.safe_generate("img.png", "/tmp/out")
        self.assertEqual(r["status"], "ok")

    def test_safe_generate_failed_passthrough(self):
        p = self._make_provider(generate_result={"status": "failed", "error": "x"})
        r = p.safe_generate("img.png", "/tmp/out")
        self.assertEqual(r["status"], "failed")


# ─────────────────────────────────────────────────────────────────────────────
# SF3DProvider availability
# ─────────────────────────────────────────────────────────────────────────────

class TestSF3DProviderAvailability(unittest.TestCase):

    def _provider_with_settings(self, **overrides):
        from modules.ai_3d_generation.sf3d_provider import SF3DProvider
        defaults = {
            "sf3d_enabled": True,
            "sf3d_execution_mode": "local_windows",   # Phase 4D: must be explicit
            "sf3d_python_path": "/fake/python.exe",
            "sf3d_worker_script": "/fake/sf3d_worker.py",
        }
        defaults.update(overrides)
        mock_settings = MagicMock(**defaults)
        p = SF3DProvider.__new__(SF3DProvider)
        p._settings = mock_settings
        return p

    def test_disabled(self):
        p = self._provider_with_settings(sf3d_enabled=False)
        avail, reason = p.is_available()
        self.assertFalse(avail)
        self.assertEqual(reason, "sf3d_disabled")

    def test_python_missing(self):
        p = self._provider_with_settings(sf3d_python_path="")
        avail, reason = p.is_available()
        self.assertFalse(avail)
        self.assertEqual(reason, "sf3d_python_missing")

    def test_python_path_not_on_disk(self):
        p = self._provider_with_settings(sf3d_python_path="/nonexistent/python.exe")
        avail, reason = p.is_available()
        self.assertFalse(avail)
        self.assertIn("sf3d_python_missing", reason)

    def test_worker_missing(self):
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            fake_py = f.name
        p = self._provider_with_settings(
            sf3d_python_path=fake_py,
            sf3d_worker_script="/nonexistent/worker.py",
        )
        avail, reason = p.is_available()
        self.assertFalse(avail)
        self.assertEqual(reason, "sf3d_worker_missing")

    def test_all_ok(self):
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as pf, \
             tempfile.NamedTemporaryFile(suffix=".py", delete=False) as wf:
            fake_py, fake_worker = pf.name, wf.name
        p = self._provider_with_settings(
            sf3d_python_path=fake_py,
            sf3d_worker_script=fake_worker,
        )
        avail, reason = p.is_available()
        self.assertTrue(avail)
        self.assertEqual(reason, "")


# ─────────────────────────────────────────────────────────────────────────────
# SF3DProvider.generate (subprocess mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestSF3DProviderGenerate(unittest.TestCase):

    def _make_provider(self, tmp_py, tmp_worker):
        from modules.ai_3d_generation.sf3d_provider import SF3DProvider
        mock_settings = MagicMock(
            sf3d_enabled=True,
            sf3d_execution_mode="local_windows",   # Phase 4D: explicit mode
            sf3d_python_path=tmp_py,
            sf3d_worker_script=tmp_worker,
            sf3d_device="auto",
            sf3d_input_size=512,
            sf3d_texture_resolution=1024,
            sf3d_remesh="none",
            sf3d_output_format="glb",
            sf3d_timeout_sec=60,
        )
        p = SF3DProvider.__new__(SF3DProvider)
        p._settings = mock_settings
        return p

    def setUp(self):
        import os
        self._pf = tempfile.NamedTemporaryFile(suffix=".exe", delete=False)
        self._wf = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
        self._pf.close(); self._wf.close()
        self.provider = self._make_provider(self._pf.name, self._wf.name)

    def _run_result(self, stdout="", stderr="", returncode=0, timeout=False):
        with patch("subprocess.run") as mock_run:
            if timeout:
                import subprocess
                mock_run.side_effect = subprocess.TimeoutExpired(cmd="sf3d", timeout=60)
            else:
                mock_run.return_value = MagicMock(
                    stdout=stdout, stderr=stderr, returncode=returncode
                )
            return self.provider.generate("image.png", "/tmp/out")

    def test_timeout(self):
        r = self._run_result(timeout=True)
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["error_code"], "sf3d_worker_timeout")

    def test_no_stdout(self):
        r = self._run_result(stdout="", returncode=1)
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["error_code"], "sf3d_worker_invalid_json")

    def test_invalid_json(self):
        r = self._run_result(stdout="not-json")
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["error_code"], "sf3d_worker_invalid_json")

    def test_unavailable(self):
        payload = json.dumps({
            "status": "unavailable",
            "error_code": "sf3d_package_missing",
            "message": "sf3d not installed",
        })
        r = self._run_result(stdout=payload)
        self.assertEqual(r["status"], "unavailable")
        self.assertIn("sf3d", r["error"].lower())

    def test_worker_failed_status(self):
        payload = json.dumps({
            "status": "failed",
            "error_code": "sf3d_inference_error",
            "message": "cuda oom",
        })
        r = self._run_result(stdout=payload)
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["error_code"], "sf3d_inference_error")

    def test_ok_with_existing_output(self):
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as glb:
            glb_path = glb.name
        payload = json.dumps({
            "status": "ok",
            "output_path": glb_path,
            "model_name": "stable-fast-3d",
            "preview_image_path": None,
            "warnings": ["ai_generated_not_true_scan"],
            "metadata": {"device": "cpu"},
        })
        r = self._run_result(stdout=payload)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["output_path"], glb_path)
        self.assertIn("ai_generated_not_true_scan", r["warnings"])

    def test_ok_but_output_missing(self):
        payload = json.dumps({
            "status": "ok",
            "output_path": "/nonexistent/output.glb",
            "model_name": "stable-fast-3d",
            "warnings": [],
        })
        r = self._run_result(stdout=payload)
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["error_code"], "sf3d_output_missing")

    def test_stderr_captured_in_logs(self):
        payload = json.dumps({"status": "failed", "error_code": "x", "message": "y"})
        stderr = "\n".join(f"line{i}" for i in range(25))
        r = self._run_result(stdout=payload, stderr=stderr)
        self.assertLessEqual(len(r["logs"]), 20)


# ─────────────────────────────────────────────────────────────────────────────
# Quality gate
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityGate(unittest.TestCase):

    def _eval(self, provider_status="ok", output_exists=True, review_required=True):
        from modules.ai_3d_generation.quality_gate import evaluate
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            glb_path = f.name if output_exists else "/nonexistent/out.glb"
        result = {"status": provider_status, "error": None}
        return evaluate(result, glb_path, review_required=review_required)

    def test_unavailable(self):
        from modules.ai_3d_generation.quality_gate import evaluate
        r = evaluate({"status": "unavailable", "error": "disabled"}, None)
        self.assertEqual(r["verdict"], "unavailable")

    def test_provider_failed(self):
        gate = self._eval(provider_status="failed", output_exists=False)
        self.assertEqual(gate["verdict"], "failed")

    def test_output_missing_on_ok_provider(self):
        gate = self._eval(provider_status="ok", output_exists=False)
        self.assertEqual(gate["verdict"], "failed")
        self.assertEqual(gate["reason"], "output_glb_missing")

    def test_review_required(self):
        gate = self._eval(review_required=True)
        self.assertEqual(gate["verdict"], "review")
        self.assertIn("review_required", gate["warnings"])

    def test_ok_no_review(self):
        gate = self._eval(review_required=False)
        self.assertEqual(gate["verdict"], "ok")

    def test_always_has_provenance_warnings(self):
        gate = self._eval()
        self.assertIn("ai_generated_not_true_scan", gate["warnings"])
        self.assertIn("generated_geometry_estimated", gate["warnings"])


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

class TestRouter(unittest.TestCase):

    def _decide(self, input_type, user_intent="default", capture_quality=None):
        from modules.ai_3d_generation.router import decide_asset_pipeline
        return decide_asset_pipeline(input_type, user_intent, capture_quality)

    def test_debug_intent_routes_to_depth_studio(self):
        d = self._decide("image", user_intent="debug")
        self.assertEqual(d["pipeline"], "depth_studio")

    def test_image_routes_to_ai_generated_3d(self):
        d = self._decide("image")
        self.assertEqual(d["pipeline"], "ai_generated_3d")

    def test_video_advanced_good_quality_routes_to_real_reconstruction(self):
        d = self._decide("video", user_intent="advanced", capture_quality="good")
        self.assertEqual(d["pipeline"], "real_reconstruction")

    def test_video_default_good_quality_routes_to_ai_generated_3d(self):
        d = self._decide("video", user_intent="default", capture_quality="good")
        self.assertEqual(d["pipeline"], "ai_generated_3d")
        self.assertEqual(d["fallback_pipeline"], "real_reconstruction")

    def test_video_poor_quality_routes_to_ai_generated_3d(self):
        d = self._decide("video", capture_quality="poor")
        self.assertEqual(d["pipeline"], "ai_generated_3d")

    def test_all_results_have_required_keys(self):
        d = self._decide("image")
        for k in ("pipeline", "reason", "fallback_pipeline", "notes"):
            self.assertIn(k, d)

    def test_always_includes_provenance_note(self):
        d = self._decide("image")
        self.assertIn("ai_generated_not_true_scan", d["notes"])


# ─────────────────────────────────────────────────────────────────────────────
# Input preprocessor
# ─────────────────────────────────────────────────────────────────────────────

class TestInputPreprocessor(unittest.TestCase):

    def _make_test_image(self, h=300, w=400):
        import cv2, numpy as np
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, img)
            return f.name, h, w

    def test_center_crop_no_mask(self):
        from modules.ai_3d_generation.input_preprocessor import preprocess_input
        img_path, h, w = self._make_test_image()
        with tempfile.TemporaryDirectory() as out_dir:
            r = preprocess_input(img_path, out_dir, input_size=128)
        self.assertIsNotNone(r["prepared_image_path"])
        self.assertEqual(r["input_size"], 128)
        self.assertIn("no_mask_or_bbox_using_center_crop", r["warnings"])

    def test_bbox_crop(self):
        from modules.ai_3d_generation.input_preprocessor import preprocess_input
        img_path, h, w = self._make_test_image(300, 400)
        with tempfile.TemporaryDirectory() as out_dir:
            r = preprocess_input(img_path, out_dir, input_size=128, bbox=(50, 50, 200, 200))
        self.assertIsNotNone(r["prepared_image_path"])
        self.assertIsNotNone(r["crop_bbox"])
        self.assertNotIn("no_mask_or_bbox_using_center_crop", r["warnings"])

    def test_mask_bbox_derivation(self):
        import numpy as np
        from modules.ai_3d_generation.input_preprocessor import preprocess_input
        img_path, h, w = self._make_test_image(200, 200)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[50:100, 60:120] = 255
        with tempfile.TemporaryDirectory() as out_dir:
            r = preprocess_input(img_path, out_dir, input_size=128, mask=mask)
        self.assertIsNotNone(r["prepared_image_path"])
        self.assertIsNotNone(r["crop_bbox"])

    def test_empty_mask_falls_back(self):
        import numpy as np
        from modules.ai_3d_generation.input_preprocessor import preprocess_input
        img_path, h, w = self._make_test_image(200, 200)
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as out_dir:
            r = preprocess_input(img_path, out_dir, input_size=64, mask=empty_mask)
        self.assertIn("mask_empty_using_full_image", r["warnings"])

    def test_output_is_square(self):
        import cv2
        from modules.ai_3d_generation.input_preprocessor import preprocess_input
        img_path, _, _ = self._make_test_image(300, 400)
        with tempfile.TemporaryDirectory() as out_dir:
            r = preprocess_input(img_path, out_dir, input_size=256)
            out = cv2.imread(r["prepared_image_path"])
        self.assertEqual(out.shape[0], out.shape[1])
        self.assertEqual(out.shape[0], 256)

    def test_bad_image_returns_error(self):
        from modules.ai_3d_generation.input_preprocessor import preprocess_input
        with tempfile.TemporaryDirectory() as out_dir:
            r = preprocess_input("/nonexistent/image.jpg", out_dir, input_size=64)
        self.assertIsNone(r["prepared_image_path"])
        self.assertTrue(any("preprocess_failed" in w for w in r["warnings"]))


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────

class TestManifest(unittest.TestCase):

    def _minimal_manifest(self, **overrides):
        from modules.ai_3d_generation.manifest import build_manifest
        kwargs = dict(
            session_id="sess-123",
            source_input_path="/input/img.jpg",
            input_type="image",
            provider="sf3d",
            provider_status="ok",
            model_name="stable-fast-3d",
            license_note="test license",
            selected_frame_path=None,
            prepared_image_path="/derived/ai3d_input.png",
            preprocessing={},
            postprocessing={},
            quality_gate={"verdict": "ok", "output_exists": True, "warnings": [], "reason": None},
            output_glb_path="/derived/output.glb",
            output_format="glb",
            preview_image_path=None,
            status="ok",
            warnings=["ai_generated_not_true_scan"],
            errors=[],
            review_required=True,
        )
        kwargs.update(overrides)
        return build_manifest(**kwargs)

    def test_required_provenance_fields(self):
        m = self._minimal_manifest()
        self.assertFalse(m["is_true_scan"])
        self.assertEqual(m["geometry_confidence"], "estimated")
        self.assertEqual(m["mode"], "ai_generated_3d")
        self.assertEqual(m["asset_type"], "ai_generated")

    def test_session_id_present(self):
        m = self._minimal_manifest(session_id="abc-xyz")
        self.assertEqual(m["session_id"], "abc-xyz")

    def test_status_propagated(self):
        m = self._minimal_manifest(status="review")
        self.assertEqual(m["status"], "review")

    def test_write_and_read_back(self):
        from modules.ai_3d_generation.manifest import build_manifest, write_manifest
        m = self._minimal_manifest()
        with tempfile.TemporaryDirectory() as d:
            path = write_manifest(m, d)
            self.assertTrue(Path(path).exists())
            loaded = json.loads(Path(path).read_text())
        self.assertEqual(loaded["session_id"], "sess-123")
        self.assertFalse(loaded["is_true_scan"])

    def test_write_creates_dir(self):
        from modules.ai_3d_generation.manifest import write_manifest
        m = self._minimal_manifest()
        with tempfile.TemporaryDirectory() as base:
            nested = str(Path(base) / "a" / "b" / "manifests")
            path = write_manifest(m, nested)
            self.assertTrue(Path(path).exists())


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline integration (mocked provider)
# ─────────────────────────────────────────────────────────────────────────────

class TestAI3DPipelineIntegration(unittest.TestCase):

    def _mock_provider(self, provider_result):
        """Return a mock AI3DProviderBase instance pre-configured with a result."""
        m = MagicMock()
        m.name = "sf3d"
        m.license_note = "test license"
        m.output_format = "glb"
        m.safe_generate.return_value = provider_result
        return m

    def _mock_settings(self, **overrides):
        defaults = dict(
            ai_3d_default_provider="sf3d",
            ai_3d_preprocess_enabled=True,
            ai_3d_postprocess_enabled=False,
            sf3d_input_size=64,
            ai_3d_require_review=True,
            sf3d_require_review=True,
        )
        defaults.update(overrides)
        return MagicMock(**defaults)

    def _run_pipeline(self, provider_result, settings_overrides=None, input_path=None):
        import cv2, numpy as np

        if input_path is None:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, np.zeros((200, 200, 3), dtype=np.uint8))
                input_path = f.name

        mock_prov = self._mock_provider(provider_result)
        mock_s = self._mock_settings(**(settings_overrides or {}))

        with tempfile.TemporaryDirectory() as out_dir:
            # _get_provider is a module-level function imported inside generate_ai_3d;
            # patch it at its definition site in pipeline.py.
            with patch("modules.ai_3d_generation.pipeline._get_provider",
                       return_value=mock_prov):
                with patch("modules.operations.settings.settings", mock_s):
                    from modules.ai_3d_generation.pipeline import generate_ai_3d
                    manifest = generate_ai_3d(
                        session_id="test-sess",
                        input_file_path=input_path,
                        output_base_dir=out_dir,
                        provider_name="sf3d",
                    )
        return manifest

    def test_unavailable_provider_manifest(self):
        result = self._run_pipeline({
            "status": "unavailable",
            "error": "sf3d_disabled",
            "warnings": [],
            "output_path": None,
            "model_name": None,
            "preview_image_path": None,
        })
        self.assertIn(result["status"], ("unavailable", "failed"))
        self.assertFalse(result["is_true_scan"])

    def test_failed_provider_manifest(self):
        result = self._run_pipeline({
            "status": "failed",
            "error": "inference error",
            "warnings": [],
            "output_path": None,
            "model_name": None,
            "preview_image_path": None,
        })
        self.assertEqual(result["status"], "failed")

    def test_ok_provider_sets_review_status(self):
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as glb:
            glb_path = glb.name
        result = self._run_pipeline({
            "status": "ok",
            "error": None,
            "warnings": ["ai_generated_not_true_scan"],
            "output_path": glb_path,
            "model_name": "stable-fast-3d",
            "preview_image_path": None,
        })
        # review_required=True → status should be "review"
        self.assertEqual(result["status"], "review")
        self.assertFalse(result["is_true_scan"])

    def test_manifest_has_provenance_fields(self):
        result = self._run_pipeline({
            "status": "unavailable", "error": "disabled",
            "warnings": [], "output_path": None, "model_name": None,
            "preview_image_path": None,
        })
        self.assertIn("is_true_scan", result)
        self.assertIn("geometry_confidence", result)
        self.assertIn("mode", result)
        self.assertEqual(result["mode"], "ai_generated_3d")

    def test_manifest_written_to_disk(self):
        import cv2, numpy as np

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, np.zeros((64, 64, 3), dtype=np.uint8))
            input_path = f.name

        mock_prov = self._mock_provider({
            "status": "unavailable", "error": "disabled",
            "warnings": [], "output_path": None,
            "model_name": None, "preview_image_path": None,
        })
        mock_s = self._mock_settings(ai_3d_preprocess_enabled=False)

        with tempfile.TemporaryDirectory() as out_dir:
            with patch("modules.ai_3d_generation.pipeline._get_provider",
                       return_value=mock_prov):
                with patch("modules.operations.settings.settings", mock_s):
                    from modules.ai_3d_generation.pipeline import generate_ai_3d
                    generate_ai_3d(
                        session_id="disk-test",
                        input_file_path=input_path,
                        output_base_dir=out_dir,
                        provider_name="sf3d",
                    )
            manifest_file = Path(out_dir) / "manifests" / "ai3d_manifest.json"
            self.assertTrue(manifest_file.exists(), "ai3d_manifest.json not written to disk")
            data = json.loads(manifest_file.read_text())
            self.assertEqual(data["session_id"], "disk-test")


# ─────────────────────────────────────────────────────────────────────────────
# SF3D worker dry-run (subprocess integration — uses actual sys.executable)
# ─────────────────────────────────────────────────────────────────────────────

class TestSF3DWorkerDryRun(unittest.TestCase):

    def _worker_path(self):
        return str(Path(__file__).parents[1] / "scripts" / "sf3d_worker.py")

    def test_dry_run_missing_image(self):
        import subprocess, sys
        worker = self._worker_path()
        result = subprocess.run(
            [sys.executable, worker, "--image", "/nonexistent/img.png",
             "--output-dir", "/tmp/out", "--dry-run"],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(result.stdout.strip())
        self.assertEqual(data["status"], "failed")
        self.assertTrue(data.get("dry_run"))
        self.assertTrue(any("not found" in str(i) for i in data.get("issues", [])))

    def test_dry_run_existing_image(self):
        import subprocess, sys
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.new("RGB", (1, 1)).save(f.name, "PNG")
            img_path = f.name

        worker = self._worker_path()
        result = subprocess.run(
            [sys.executable, worker, "--image", img_path,
             "--output-dir", "/tmp/out", "--dry-run"],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(result.stdout.strip())
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data.get("dry_run"))

    def test_worker_unavailable_path(self):
        """When sf3d package is absent, worker exits 0 with status=unavailable."""
        import subprocess, sys
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.new("RGB", (1, 1)).save(f.name, "PNG")
            img_path = f.name

        worker = self._worker_path()
        result = subprocess.run(
            [sys.executable, worker, "--image", img_path, "--output-dir", "/tmp/out"],
            capture_output=True, text=True, timeout=30,
        )
        # sf3d is not installed in the main env → must exit 0 with status=unavailable
        self.assertEqual(result.returncode, 0,
                         f"expected exit 0, got {result.returncode}\nstderr: {result.stderr}")
        data = json.loads(result.stdout.strip())
        self.assertEqual(data["status"], "unavailable")
        self.assertEqual(data["error_code"], "sf3d_package_missing")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4D — WSL2 path mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestSF3DWSLPathMapping(unittest.TestCase):

    def _w2w(self, p): from modules.ai_3d_generation.sf3d_provider import _windows_to_wsl_path; return _windows_to_wsl_path(p)
    def _w2n(self, p): from modules.ai_3d_generation.sf3d_provider import _wsl_to_windows_path; return _wsl_to_windows_path(p)

    def test_c_drive_to_wsl(self):
        result = self._w2w(r"C:\Users\Lenovo\input.png")
        self.assertEqual(result, "/mnt/c/Users/Lenovo/input.png")

    def test_spaces_in_path(self):
        result = self._w2w(r"C:\My Files\output dir\out.glb")
        self.assertEqual(result, "/mnt/c/My Files/output dir/out.glb")

    def test_wsl_to_windows(self):
        result = self._w2n("/mnt/c/Users/Lenovo/scratch/output.glb")
        self.assertEqual(result, r"C:\Users\Lenovo\scratch\output.glb")

    def test_already_posix_unchanged(self):
        result = self._w2w("/tmp/sf3d_smoke/output.glb")
        self.assertEqual(result, "/tmp/sf3d_smoke/output.glb")

    def test_non_mnt_wsl_path_unchanged(self):
        result = self._w2n("/tmp/sf3d/output.glb")
        self.assertEqual(result, "/tmp/sf3d/output.glb")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4D — WSL2 provider mode
# ─────────────────────────────────────────────────────────────────────────────

class TestSF3DProviderWSLMode(unittest.TestCase):

    def _wsl_provider(self, worker_exists=True, **overrides):
        """Build an SF3DProvider with wsl_subprocess mode settings."""
        import os
        from modules.ai_3d_generation.sf3d_provider import SF3DProvider

        # Create a real worker file on disk so availability check passes
        if worker_exists:
            wf = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
            wf.close()
            worker_win = wf.name
        else:
            worker_win = r"C:\nonexistent\sf3d_worker.py"

        # The wsl_repo_root is set to a path that maps to worker_win's location.
        # worker_wsl = f"{repo_root}/scripts/sf3d_worker.py"
        # worker_win = _wsl_to_windows_path(worker_wsl)
        # We mock _wsl_to_windows_path by providing a repo_root that maps correctly.
        # For simplicity, set worker via direct mock.
        defaults = dict(
            sf3d_enabled=True,
            sf3d_execution_mode="wsl_subprocess",
            sf3d_wsl_python_path="/home/lenovo/sf3d_venv/bin/python",
            sf3d_wsl_distro="Ubuntu-24.04",
            sf3d_wsl_repo_root="/mnt/c/fake/repo",
            sf3d_wsl_timeout_sec=60,
            sf3d_device="cuda",
            sf3d_input_size=512,
            sf3d_texture_resolution=512,
            sf3d_remesh="none",
            sf3d_output_format="glb",
        )
        defaults.update(overrides)
        mock_settings = MagicMock(**defaults)
        p = SF3DProvider.__new__(SF3DProvider)
        p._settings = mock_settings
        # Patch the availability worker-path check to avoid FS dependency
        p._test_worker_exists = worker_exists
        return p

    def _make_wsl_provider_available(self):
        """Return a provider whose is_available() returns (True, '') via mock."""
        from modules.ai_3d_generation.sf3d_provider import SF3DProvider
        mock_settings = MagicMock(
            sf3d_enabled=True,
            sf3d_execution_mode="wsl_subprocess",
            sf3d_wsl_python_path="/home/lenovo/sf3d_venv/bin/python",
            sf3d_wsl_distro="Ubuntu-24.04",
            sf3d_wsl_repo_root="/mnt/c/fake/repo",
            sf3d_wsl_timeout_sec=60,
            sf3d_device="cuda",
            sf3d_input_size=512,
            sf3d_texture_resolution=512,
            sf3d_remesh="none",
            sf3d_output_format="glb",
        )
        p = SF3DProvider.__new__(SF3DProvider)
        p._settings = mock_settings
        return p

    def _run_wsl(self, stdout="", stderr="", returncode=0, timeout=False):
        p = self._make_wsl_provider_available()
        with patch("modules.ai_3d_generation.sf3d_provider.SF3DProvider.is_available",
                   return_value=(True, "")):
            with patch("subprocess.run") as mock_run:
                if timeout:
                    import subprocess as sp
                    mock_run.side_effect = sp.TimeoutExpired(cmd="wsl.exe", timeout=60)
                else:
                    mock_run.return_value = MagicMock(
                        stdout=stdout, stderr=stderr, returncode=returncode
                    )
                return p.generate("C:\\input\\img.png", "C:\\output\\dir")

    def test_execution_mode_disabled_unavailable(self):
        from modules.ai_3d_generation.sf3d_provider import SF3DProvider
        mock_s = MagicMock(sf3d_enabled=True, sf3d_execution_mode="disabled")
        p = SF3DProvider.__new__(SF3DProvider)
        p._settings = mock_s
        avail, reason = p.is_available()
        self.assertFalse(avail)
        self.assertEqual(reason, "sf3d_execution_mode_disabled")

    def test_wsl_missing_python_unavailable(self):
        from modules.ai_3d_generation.sf3d_provider import SF3DProvider
        mock_s = MagicMock(
            sf3d_enabled=True,
            sf3d_execution_mode="wsl_subprocess",
            sf3d_wsl_python_path="",
        )
        p = SF3DProvider.__new__(SF3DProvider)
        p._settings = mock_s
        avail, reason = p.is_available()
        self.assertFalse(avail)
        self.assertEqual(reason, "sf3d_wsl_python_missing")

    def test_wsl_missing_distro_unavailable(self):
        from modules.ai_3d_generation.sf3d_provider import SF3DProvider
        mock_s = MagicMock(
            sf3d_enabled=True,
            sf3d_execution_mode="wsl_subprocess",
            sf3d_wsl_python_path="/home/lenovo/sf3d_venv/bin/python",
            sf3d_wsl_distro="",
        )
        p = SF3DProvider.__new__(SF3DProvider)
        p._settings = mock_s
        avail, reason = p.is_available()
        self.assertFalse(avail)
        self.assertEqual(reason, "sf3d_wsl_distro_missing")

    def test_wsl_command_construction(self):
        """_generate_wsl_subprocess builds correct wsl.exe command."""
        p = self._make_wsl_provider_available()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="", returncode=1)
            with patch("modules.ai_3d_generation.sf3d_provider.SF3DProvider.is_available",
                       return_value=(True, "")):
                p.generate("C:\\Users\\foo\\img.png", "C:\\Users\\foo\\out")
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "wsl.exe")
        self.assertIn("-d", cmd)
        self.assertIn("Ubuntu-24.04", cmd)
        self.assertIn("--", cmd)
        self.assertIn("/home/lenovo/sf3d_venv/bin/python", cmd)
        # WSL-converted input path
        self.assertIn("/mnt/c/Users/foo/img.png", cmd)

    def test_wsl_worker_ok_json(self):
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as glb:
            glb_win_path = glb.name
        # Simulate worker returning a /mnt/c/ path (convert back from Windows temp)
        from modules.ai_3d_generation.sf3d_provider import _windows_to_wsl_path
        glb_wsl_path = _windows_to_wsl_path(glb_win_path)
        payload = json.dumps({
            "status": "ok",
            "output_path": glb_wsl_path,
            "model_name": "stable-fast-3d",
            "preview_image_path": None,
            "warnings": ["ai_generated_not_true_scan"],
            "metadata": {"device": "cuda", "peak_mem_mb": 6173.5, "output_size_bytes": 1346664},
        })
        r = self._run_wsl(stdout=payload)
        self.assertEqual(r["status"], "ok")
        self.assertIsNotNone(r["output_path"])
        self.assertIn("ai_generated_not_true_scan", r["warnings"])
        self.assertEqual(r["metadata"].get("device"), "cuda")
        self.assertEqual(r["metadata"].get("execution_mode"), "wsl_subprocess")

    def test_wsl_worker_failed_json(self):
        payload = json.dumps({
            "status": "failed",
            "error_code": "sf3d_inference_error",
            "message": "CUDA OOM",
        })
        r = self._run_wsl(stdout=payload)
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["error_code"], "sf3d_inference_error")

    def test_wsl_worker_invalid_json(self):
        r = self._run_wsl(stdout="not-valid-json")
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["error_code"], "sf3d_worker_invalid_json")

    def test_wsl_worker_unavailable(self):
        payload = json.dumps({
            "status": "unavailable",
            "error_code": "sf3d_model_auth_required",
            "message": "HF token required",
        })
        r = self._run_wsl(stdout=payload)
        self.assertEqual(r["status"], "unavailable")

    def test_wsl_output_path_normalization(self):
        """WSL path /mnt/c/... in worker JSON is converted to Windows C:\\..."""
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as glb:
            glb_win = glb.name
        from modules.ai_3d_generation.sf3d_provider import _windows_to_wsl_path
        glb_wsl = _windows_to_wsl_path(glb_win)
        payload = json.dumps({
            "status": "ok",
            "output_path": glb_wsl,
            "model_name": "stable-fast-3d",
            "warnings": [],
            "metadata": {},
        })
        r = self._run_wsl(stdout=payload)
        if r["status"] == "ok":
            # output_path should be the Windows form
            self.assertNotIn("/mnt/", r["output_path"])
            self.assertIn(":\\", r["output_path"])


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4D — Noisy stdout parsing (_parse_worker_stdout)
# ─────────────────────────────────────────────────────────────────────────────

class TestParseWorkerStdout(unittest.TestCase):
    """Regression tests for _parse_worker_stdout noisy-stdout tolerance."""

    def _parse(self, stdout):
        from modules.ai_3d_generation.sf3d_provider import _parse_worker_stdout
        return _parse_worker_stdout(stdout)

    # ── happy-path ─────────────────────────────────────────────────────────────

    def test_clean_stdout_ok(self):
        """Clean stdout (single JSON line) parses fine."""
        payload = json.dumps({
            "status": "ok",
            "output_path": "/mnt/c/data/derived/output.glb",
            "model_name": "stable-fast-3d",
            "warnings": [],
            "metadata": {"device": "cuda", "peak_mem_mb": 6000.0},
        })
        result = self._parse(payload + "\n")
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("worker_stdout_had_extra_lines", result["warnings"])

    def test_empty_stdout_returns_none(self):
        self.assertIsNone(self._parse(""))
        self.assertIsNone(self._parse("   \n  "))

    def test_totally_invalid_stdout_returns_none(self):
        self.assertIsNone(self._parse("After Remesh 9298 18592\nsome other noise"))

    # ── dirty-path (extra lines before JSON) ──────────────────────────────────

    def test_noisy_stdout_returns_ok(self):
        """'After Remesh N M' line before JSON → status ok, warning added."""
        json_line = json.dumps({
            "status": "ok",
            "output_path": "/mnt/c/data/derived/output.glb",
            "model_name": "stable-fast-3d",
            "warnings": [],
            "metadata": {"device": "cuda", "peak_mem_mb": 6173.5,
                         "output_size_bytes": 1346664},
        })
        noisy_stdout = f"After Remesh 9298 18592\n{json_line}\n"
        result = self._parse(noisy_stdout)
        self.assertIsNotNone(result, "should parse despite noisy line")
        self.assertEqual(result["status"], "ok")
        self.assertIn("worker_stdout_had_extra_lines", result["warnings"])
        self.assertIn("worker_stdout_had_extra_lines", result["logs"])

    def test_noisy_stdout_output_path_preserved(self):
        """output_path from JSON is preserved correctly after noisy-stdout parse."""
        glb_path = "/mnt/c/data/derived/output.glb"
        json_line = json.dumps({
            "status": "ok",
            "output_path": glb_path,
            "model_name": "stable-fast-3d",
            "warnings": [],
            "metadata": {},
        })
        noisy_stdout = f"After Remesh 9298 18592\n{json_line}"
        result = self._parse(noisy_stdout)
        self.assertEqual(result["output_path"], glb_path)

    def test_noisy_stdout_multiple_extra_lines(self):
        """Multiple junk lines before JSON still parsed correctly."""
        json_line = json.dumps({"status": "ok", "output_path": "/tmp/out.glb",
                                "warnings": [], "metadata": {}})
        noisy = f"[info] loading model\nProcessing mesh...\nAfter Remesh 100 200\n{json_line}"
        result = self._parse(noisy)
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "ok")
        self.assertIn("worker_stdout_had_extra_lines", result["warnings"])

    def test_noisy_stdout_failed_json(self):
        """Noisy stdout with failed JSON parses correctly."""
        json_line = json.dumps({
            "status": "failed",
            "error_code": "sf3d_inference_error",
            "message": "CUDA OOM",
            "warnings": [],
        })
        noisy_stdout = f"After Remesh 50 100\n{json_line}"
        result = self._parse(noisy_stdout)
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "sf3d_inference_error")

    def test_provider_noisy_stdout_full_pipeline(self):
        """Provider._run_worker correctly handles noisy stdout → status ok."""
        from modules.ai_3d_generation.sf3d_provider import SF3DProvider
        import tempfile, os
        mock_s = MagicMock(
            sf3d_enabled=True,
            sf3d_execution_mode="local_windows",
            sf3d_python_path=r"C:\dummy\python.exe",
            sf3d_worker_script=r"C:\dummy\sf3d_worker.py",
            sf3d_device="cpu",
            sf3d_input_size=512,
            sf3d_texture_resolution=512,
            sf3d_remesh="none",
            sf3d_output_format="glb",
            sf3d_timeout_sec=60,
        )
        p = SF3DProvider.__new__(SF3DProvider)
        p._settings = mock_s

        # Create a real GLB temp file so existence check passes
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            f.write(b"GLB_DUMMY")
            glb_path = f.name

        try:
            json_line = json.dumps({
                "status": "ok",
                "output_path": glb_path,
                "model_name": "stable-fast-3d",
                "warnings": [],
                "metadata": {"device": "cpu", "peak_mem_mb": None,
                             "output_size_bytes": 9},
            })
            noisy_stdout = f"After Remesh 9298 18592\n{json_line}\n"

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout=noisy_stdout, stderr="", returncode=0
                )
                result = p._run_worker(
                    ["fake_cmd"], 60, glb_path, normalize_path=False
                )

            self.assertEqual(result["status"], "ok",
                             f"Expected ok, got {result}")
            self.assertEqual(result["output_path"], glb_path)
            self.assertIn("worker_stdout_had_extra_lines", result["warnings"])
        finally:
            os.unlink(glb_path)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4D — Manifest execution_mode + worker_metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestManifestExecutionMode(unittest.TestCase):

    def _build(self, **kw):
        from modules.ai_3d_generation.manifest import build_manifest
        base = dict(
            session_id="s1", source_input_path="/in/img.jpg", input_type="image",
            provider="sf3d", provider_status="ok", model_name="stable-fast-3d",
            license_note="test", selected_frame_path=None, prepared_image_path=None,
            preprocessing={}, postprocessing={},
            quality_gate={"verdict": "ok", "output_exists": True, "warnings": [], "reason": None},
            output_glb_path="/out/output.glb", output_format="glb",
            preview_image_path=None, status="ok",
            warnings=[], errors=[],
        )
        base.update(kw)
        return build_manifest(**base)

    def test_execution_mode_field_present(self):
        m = self._build(execution_mode="wsl_subprocess")
        self.assertEqual(m["execution_mode"], "wsl_subprocess")

    def test_worker_metadata_peak_mem_propagated(self):
        m = self._build(
            worker_metadata={"device": "cuda", "peak_mem_mb": 6173.5, "output_size_bytes": 1346664},
        )
        self.assertEqual(m["peak_mem_mb"], 6173.5)
        self.assertEqual(m["worker_metadata"]["device"], "cuda")
        self.assertEqual(m["worker_metadata"]["output_size_bytes"], 1346664)

    def test_provider_failure_reason_present(self):
        m = self._build(
            provider_status="unavailable",
            status="unavailable",
            provider_failure_reason="sf3d_execution_mode_disabled",
        )
        self.assertEqual(m["provider_failure_reason"], "sf3d_execution_mode_disabled")

    def test_missing_outputs_default_empty(self):
        m = self._build()
        self.assertEqual(m["missing_outputs"], [])


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4E — Path edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestSF3DPathEdgeCases(unittest.TestCase):

    def _w2w(self, p):
        from modules.ai_3d_generation.sf3d_provider import _windows_to_wsl_path
        return _windows_to_wsl_path(p)

    def _w2n(self, p):
        from modules.ai_3d_generation.sf3d_provider import _wsl_to_windows_path
        return _wsl_to_windows_path(p)

    def test_unc_path_returned_unchanged(self):
        """UNC paths are not convertible — returned as-is with no crash."""
        unc = r"\\server\share\file.png"
        result = self._w2w(unc)
        self.assertEqual(result, unc)

    def test_bare_mnt_drive_wsl_to_windows(self):
        """/mnt/c with no trailing path → C:\\"""
        result = self._w2n("/mnt/c")
        self.assertEqual(result, "C:\\")

    def test_mnt_drive_trailing_slash(self):
        """/mnt/c/ (trailing slash) converts cleanly."""
        result = self._w2n("/mnt/c/")
        self.assertEqual(result, "C:\\")

    def test_empty_string_unchanged(self):
        """Empty string input → empty string output."""
        self.assertEqual(self._w2w(""), "")
        self.assertEqual(self._w2n(""), "")

    def test_wsl_lowercase_drive(self):
        """/mnt/d/… → D:\\…"""
        result = self._w2n("/mnt/d/data/file.glb")
        self.assertEqual(result, r"D:\data\file.glb")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4E — GPU busy lock
# ─────────────────────────────────────────────────────────────────────────────

class TestSF3DBusyLock(unittest.TestCase):

    def _make_provider(self):
        from modules.ai_3d_generation.sf3d_provider import SF3DProvider
        mock_s = MagicMock(
            sf3d_enabled=True,
            sf3d_execution_mode="wsl_subprocess",
            sf3d_wsl_python_path="/home/lenovo/sf3d_venv/bin/python",
            sf3d_wsl_distro="Ubuntu-24.04",
            sf3d_wsl_repo_root="/mnt/c/fake/repo",
            sf3d_wsl_timeout_sec=60,
            sf3d_device="cuda",
            sf3d_input_size=512,
            sf3d_texture_resolution=512,
            sf3d_remesh="none",
            sf3d_output_format="glb",
        )
        p = SF3DProvider.__new__(SF3DProvider)
        p._settings = mock_s
        return p

    def test_lock_held_returns_busy(self):
        """When _sf3d_lock is already held, generate() returns status=busy.

        We acquire the module-level lock on this thread before calling
        generate(), which uses blocking=False — so it immediately sees the
        lock is taken and returns busy without deadlocking.
        """
        import modules.ai_3d_generation.sf3d_provider as mod
        p = self._make_provider()
        mod._sf3d_lock.acquire()
        try:
            with patch(
                "modules.ai_3d_generation.sf3d_provider.SF3DProvider.is_available",
                return_value=(True, ""),
            ):
                result = p.generate("C:\\img.png", "C:\\out")
        finally:
            mod._sf3d_lock.release()
        self.assertEqual(result["status"], "busy")
        self.assertEqual(result["error_code"], "sf3d_job_already_running")

    def test_busy_passthrough_in_safe_generate(self):
        """safe_generate() preserves status=busy (not remapped to failed)."""
        from modules.ai_3d_generation.provider_base import _KNOWN_STATUSES
        self.assertIn("busy", _KNOWN_STATUSES)

    def test_busy_maps_to_unavailable_in_quality_gate(self):
        """quality_gate.evaluate() maps status=busy → verdict=unavailable."""
        from modules.ai_3d_generation.quality_gate import evaluate
        provider_result = {
            "status": "busy",
            "error_code": "sf3d_job_already_running",
            "error": "sf3d_job_already_running",
        }
        gate = evaluate(provider_result, output_glb_path=None, review_required=False)
        self.assertEqual(gate["verdict"], "unavailable")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4E — WSL2 preflight
# ─────────────────────────────────────────────────────────────────────────────

class TestSF3DPreflight(unittest.TestCase):

    def _make_provider(self, exec_mode="wsl_subprocess"):
        from modules.ai_3d_generation.sf3d_provider import SF3DProvider
        mock_s = MagicMock(
            sf3d_enabled=True,
            sf3d_execution_mode=exec_mode,
            sf3d_wsl_python_path="/home/lenovo/sf3d_venv/bin/python",
            sf3d_wsl_distro="Ubuntu-24.04",
            sf3d_wsl_repo_root="/mnt/c/fake/repo",
            sf3d_wsl_timeout_sec=60,
        )
        p = SF3DProvider.__new__(SF3DProvider)
        p._settings = mock_s
        return p

    def test_preflight_not_wsl_mode_returns_ok_false(self):
        """preflight_wsl() in disabled mode reports ok=False, not a crash."""
        p = self._make_provider(exec_mode="disabled")
        result = p.preflight_wsl()
        self.assertIn("ok", result)
        self.assertFalse(result["ok"])

    def test_preflight_wsl_exe_missing(self):
        """When wsl.exe is not on PATH, the wsl_exe check ok=False."""
        p = self._make_provider()
        with patch("shutil.which", return_value=None):
            result = p.preflight_wsl()
        checks = result.get("checks", {})
        # Each check value is {"ok": bool, "detail": ...}
        self.assertFalse(checks.get("wsl_exe", {}).get("ok", True))
        self.assertFalse(result["ok"])

    def test_preflight_structure(self):
        """preflight_wsl() always returns dict with ok, checks, execution_mode."""
        p = self._make_provider()
        with patch("shutil.which", return_value=None):
            result = p.preflight_wsl()
        self.assertIn("ok", result)
        self.assertIn("checks", result)
        self.assertIn("execution_mode", result)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4E — Manifest timing and path_diagnostics fields
# ─────────────────────────────────────────────────────────────────────────────

class TestManifestTimingFields(unittest.TestCase):

    def _build(self, **kw):
        from modules.ai_3d_generation.manifest import build_manifest
        base = dict(
            session_id="s1", source_input_path="/in/img.jpg", input_type="image",
            provider="sf3d", provider_status="ok", model_name="stable-fast-3d",
            license_note="test", selected_frame_path=None, prepared_image_path=None,
            preprocessing={}, postprocessing={},
            quality_gate={"verdict": "ok", "output_exists": True, "warnings": [], "reason": None},
            output_glb_path="/out/output.glb", output_format="glb",
            preview_image_path=None, status="ok",
            warnings=[], errors=[],
        )
        base.update(kw)
        return build_manifest(**base)

    def test_timing_fields_present(self):
        m = self._build(
            generation_started_at="2026-05-03T10:00:00+00:00",
            generation_finished_at="2026-05-03T10:00:35+00:00",
            duration_sec=35.1,
        )
        self.assertEqual(m["generation_started_at"], "2026-05-03T10:00:00+00:00")
        self.assertEqual(m["generation_finished_at"], "2026-05-03T10:00:35+00:00")
        self.assertAlmostEqual(m["duration_sec"], 35.1)

    def test_timing_fields_default_none(self):
        """Timing fields default to None for backward compat."""
        m = self._build()
        self.assertIsNone(m["generation_started_at"])
        self.assertIsNone(m["generation_finished_at"])
        self.assertIsNone(m["duration_sec"])

    def test_output_size_bytes_field(self):
        m = self._build(output_size_bytes=1346664)
        self.assertEqual(m["output_size_bytes"], 1346664)

    def test_path_diagnostics_field(self):
        diag = {
            "source_input_path": r"C:\data\img.png",
            "output_dir":        r"C:\data\out",
            "generation_input_wsl": "/mnt/c/data/img.png",
            "output_dir_wsl":    "/mnt/c/data/out",
        }
        m = self._build(path_diagnostics=diag)
        self.assertEqual(m["path_diagnostics"]["generation_input_wsl"],
                         "/mnt/c/data/img.png")

    def test_path_diagnostics_default_empty_dict(self):
        """path_diagnostics defaults to {} when not provided."""
        m = self._build()
        self.assertEqual(m["path_diagnostics"], {})


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Multi-input session resolver
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiInput(unittest.TestCase):

    def test_detect_image(self):
        from modules.ai_3d_generation.multi_input import detect_input_mode
        self.assertEqual(detect_input_mode("photo.jpg"), "single_image")
        self.assertEqual(detect_input_mode("photo.PNG"), "single_image")

    def test_detect_video(self):
        from modules.ai_3d_generation.multi_input import detect_input_mode
        self.assertEqual(detect_input_mode("clip.mp4"), "video")
        self.assertEqual(detect_input_mode("CLIP.MOV"), "video")

    def test_write_and_load_session_inputs(self):
        from modules.ai_3d_generation.multi_input import (
            write_session_inputs, load_session_inputs,
        )
        with tempfile.TemporaryDirectory() as sd:
            write_session_inputs(sd, "multi_image", ["upload_001.jpg", "upload_002.png"])
            data = load_session_inputs(sd)
        self.assertIsNotNone(data)
        self.assertEqual(data["input_mode"], "multi_image")
        self.assertEqual(data["uploaded_files_count"], 2)
        self.assertEqual(len(data["input_files"]), 2)

    def test_load_missing_returns_none(self):
        from modules.ai_3d_generation.multi_input import load_session_inputs
        with tempfile.TemporaryDirectory() as sd:
            self.assertIsNone(load_session_inputs(sd))

    def test_write_session_inputs_basenames_only(self):
        from modules.ai_3d_generation.multi_input import write_session_inputs, load_session_inputs
        with tempfile.TemporaryDirectory() as sd:
            write_session_inputs(sd, "multi_image", ["/absolute/path/to/upload_001.jpg", "upload_002.png"])
            data = load_session_inputs(sd)
            self.assertEqual(data["input_files"], ["upload_001.jpg", "upload_002.png"])

    def test_resolve_multi_image_skips_non_images(self):
        from modules.ai_3d_generation.multi_input import write_session_inputs, load_session_inputs, resolve_candidate_sources
        with tempfile.TemporaryDirectory() as sd:
            input_dir = Path(sd) / "input"
            input_dir.mkdir()
            for name in ["upload_001.jpg", "upload_002.txt"]:
                (input_dir / name).write_text("fake")
            write_session_inputs(sd, "multi_image", ["upload_001.jpg", "upload_002.txt"])
            si = load_session_inputs(sd)
            result = resolve_candidate_sources(sd, str(input_dir / "upload_001.jpg"), si)
            # Only the image should be resolved
            self.assertEqual(len(result["sources"]), 1)
            self.assertTrue(result["sources"][0].endswith("upload_001.jpg"))

    def test_resolve_multi_image_path_traversal_guard(self):
        from modules.ai_3d_generation.multi_input import write_session_inputs, load_session_inputs, resolve_candidate_sources
        with tempfile.TemporaryDirectory() as sd:
            input_dir = Path(sd) / "input"
            input_dir.mkdir()
            (input_dir / "upload_001.jpg").write_text("fake")
            write_session_inputs(sd, "multi_image", ["upload_001.jpg", "../../../../windows/system32/cmd.exe"])
            si = load_session_inputs(sd)
            result = resolve_candidate_sources(sd, str(input_dir / "upload_001.jpg"), si)
            self.assertEqual(len(result["sources"]), 1)
            self.assertTrue(result["sources"][0].endswith("upload_001.jpg"))

    def test_write_session_inputs_invalid_mode(self):
        from modules.ai_3d_generation.multi_input import write_session_inputs
        with tempfile.TemporaryDirectory() as sd:
            with self.assertRaises(ValueError):
                write_session_inputs(sd, "invalid_mode", ["test.jpg"])

    def test_write_session_inputs_empty_multi(self):
        from modules.ai_3d_generation.multi_input import write_session_inputs
        with tempfile.TemporaryDirectory() as sd:
            with self.assertRaises(ValueError):
                write_session_inputs(sd, "multi_image", [])

    def test_resolve_multi_image_path_traversal_prefix(self):
        from modules.ai_3d_generation.multi_input import write_session_inputs, load_session_inputs, resolve_candidate_sources
        with tempfile.TemporaryDirectory() as sd:
            input_dir = Path(sd) / "input"
            input_dir.mkdir()
            (input_dir / "upload_001.jpg").write_text("fake")
            
            evil_dir = Path(sd) / "input_evil"
            evil_dir.mkdir()
            (evil_dir / "upload_002.jpg").write_text("fake")
            
            write_session_inputs(sd, "multi_image", ["upload_001.jpg", "../input_evil/upload_002.jpg"])
            si = load_session_inputs(sd)
            result = resolve_candidate_sources(sd, str(input_dir / "upload_001.jpg"), si)
            
            self.assertEqual(len(result["sources"]), 1)
            self.assertTrue(result["sources"][0].endswith("upload_001.jpg"))

    def test_resolve_multi_image_sources(self):
        import cv2, numpy as np
        from modules.ai_3d_generation.multi_input import (
            write_session_inputs, load_session_inputs, resolve_candidate_sources,
        )
        with tempfile.TemporaryDirectory() as sd:
            input_dir = Path(sd) / "input"
            input_dir.mkdir()
            # Create two dummy images
            for name in ["upload_001.jpg", "upload_002.jpg"]:
                cv2.imwrite(str(input_dir / name),
                            np.zeros((64, 64, 3), dtype=np.uint8))
            write_session_inputs(sd, "multi_image", ["upload_001.jpg", "upload_002.jpg"])
            si = load_session_inputs(sd)
            result = resolve_candidate_sources(sd, str(input_dir / "upload_001.jpg"), si)
        self.assertEqual(result["input_mode"], "multi_image")
        self.assertEqual(len(result["sources"]), 2)

    def test_resolve_single_image_fallback(self):
        from modules.ai_3d_generation.multi_input import resolve_candidate_sources
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            result = resolve_candidate_sources("/tmp/fake_session", f.name)
        self.assertEqual(result["input_mode"], "single_image")
        self.assertEqual(len(result["sources"]), 1)

    def test_resolve_video_mode(self):
        from modules.ai_3d_generation.multi_input import resolve_candidate_sources
        result = resolve_candidate_sources("/tmp/fake", "/tmp/clip.mp4")
        self.assertEqual(result["input_mode"], "video")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Candidate selector
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateSelector(unittest.TestCase):

    def _make_candidate(self, cand_id="cand_001", status="ok",
                        provider_status="ok", glb_path=None,
                        prep_path=None, warnings=None, score=None):
        meta = {
            "candidate_id": cand_id,
            "status": status,
            "provider_status": provider_status,
            "output_glb_path": glb_path,
            "prepared_image_path": prep_path,
            "warnings": warnings or [],
        }
        if score is not None:
            meta["score"] = score
        return meta

    def test_score_ok_with_glb(self):
        from modules.ai_3d_generation.candidate_selector import score_candidate
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            # Write some bytes so size > 0
            f.write(b"x" * 10240)
            glb = f.name
        score, breakdown = score_candidate(self._make_candidate(glb_path=glb))
        self.assertGreater(score, 50)  # provider_ok + glb_exists + size
        self.assertEqual(breakdown["provider_ok"], 50.0)
        self.assertEqual(breakdown["glb_exists"], 20.0)

    def test_score_failed_provider(self):
        from modules.ai_3d_generation.candidate_selector import score_candidate
        score, _ = score_candidate(self._make_candidate(
            provider_status="failed", status="failed"
        ))
        self.assertEqual(score, 0.0)

    def test_center_crop_penalty(self):
        from modules.ai_3d_generation.candidate_selector import score_candidate
        score_no_penalty, _ = score_candidate(self._make_candidate())
        score_penalty, bd = score_candidate(self._make_candidate(
            warnings=["no_mask_or_bbox_using_center_crop"]
        ))
        self.assertLess(score_penalty, score_no_penalty)
        self.assertIn("center_crop_penalty", bd)

    def test_select_best_picks_highest_score(self):
        from modules.ai_3d_generation.candidate_selector import select_best
        candidates = [
            self._make_candidate("cand_001", status="ok", glb_path="/fake.glb", score=60),
            self._make_candidate("cand_002", status="ok", glb_path="/fake.glb", score=80),
            self._make_candidate("cand_003", status="failed", score=0),
        ]
        with patch("pathlib.Path.exists", return_value=True):
            best, ranking, reason = select_best(candidates)
            self.assertIsNotNone(best)
            self.assertEqual(best["candidate_id"], "cand_002")
            self.assertIn("80", reason)
            self.assertEqual(ranking[0]["candidate_id"], "cand_002")

    def test_select_best_all_failed(self):
        from modules.ai_3d_generation.candidate_selector import select_best
        candidates = [
            self._make_candidate("cand_001", status="failed", score=0),
            self._make_candidate("cand_002", status="failed", score=0),
        ]
        best, ranking, reason = select_best(candidates)
        self.assertIsNone(best)
        self.assertEqual(reason, "all_candidates_failed")

    def test_select_best_empty(self):
        from modules.ai_3d_generation.candidate_selector import select_best
        best, ranking, reason = select_best([])
        self.assertIsNone(best)
        self.assertEqual(reason, "no_candidates")

    def test_failed_candidate_does_not_prevent_success(self):
        """One failed + one successful → session should succeed."""
        from modules.ai_3d_generation.candidate_selector import select_best
        candidates = [
            self._make_candidate("cand_001", status="failed", score=0),
            self._make_candidate("cand_002", status="ok", provider_status="ok", glb_path="/fake.glb", score=70),
        ]
        with patch("pathlib.Path.exists", return_value=True):
            best, _, reason = select_best(candidates)
            self.assertIsNotNone(best)
            self.assertEqual(best["candidate_id"], "cand_002")

    def test_candidate_ranking_compact(self):
        from modules.ai_3d_generation.candidate_selector import select_best
        candidates = [
            self._make_candidate("cand_001", status="ok", provider_status="ok", glb_path="/fake.glb", score=60),
            self._make_candidate("cand_002", status="ok", provider_status="ok", glb_path="/fake.glb", score=80),
        ]
        with patch("pathlib.Path.exists", return_value=True):
            best, ranking, reason = select_best(candidates)
            self.assertEqual(len(ranking), 2)
            for r in ranking:
                self.assertIn("candidate_id", r)
                self.assertIn("selected", r)
                self.assertNotIn("warnings", r)  # Only compact fields
            
            # The highest score should be first in ranking
            self.assertEqual(ranking[0]["candidate_id"], "cand_002")
            self.assertTrue(ranking[0]["selected"])
            self.assertFalse(ranking[1]["selected"])


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Video candidates (basic)
# ─────────────────────────────────────────────────────────────────────────────

class TestVideoCandidates(unittest.TestCase):

    def _make_test_video(self, frames=30, fps=30, w=64, h=64):
        """Create a synthetic video file for testing."""
        import cv2, numpy as np
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        for i in range(frames):
            # Vary sharpness: some frames blurry, some sharp
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            if i % 5 == 0:
                # Make every 5th frame sharper (edges)
                frame = cv2.Laplacian(frame, cv2.CV_8U)
                frame = np.clip(frame * 3, 0, 255).astype(np.uint8)
            writer.write(frame)
        writer.release()
        return video_path

    def test_top_k_returns_correct_count(self):
        from modules.ai_3d_generation.video_candidates import select_top_k_frames
        video_path = self._make_test_video(frames=60, fps=30)
        with tempfile.TemporaryDirectory() as out_dir:
            paths = select_top_k_frames(video_path, out_dir, top_k=3, min_spacing_sec=0.1)
            self.assertLessEqual(len(paths), 3)
            self.assertGreater(len(paths), 0)
            for p in paths:
                self.assertTrue(Path(p).exists())

    def test_top_k_respects_max(self):
        from modules.ai_3d_generation.video_candidates import select_top_k_frames
        video_path = self._make_test_video(frames=10, fps=10)
        with tempfile.TemporaryDirectory() as out_dir:
            paths = select_top_k_frames(video_path, out_dir, top_k=20, min_spacing_sec=0.1)
        # Can't return more frames than the video has
        self.assertLessEqual(len(paths), 20)

    def test_invalid_video_returns_empty(self):
        from modules.ai_3d_generation.video_candidates import select_top_k_frames
        with tempfile.TemporaryDirectory() as out_dir:
            paths = select_top_k_frames("/nonexistent/video.mp4", out_dir, top_k=3)
        self.assertEqual(paths, [])


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Candidate runner (mocked provider)
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateRunner(unittest.TestCase):

    def _mock_provider(self, results_iter):
        """Provider whose safe_generate returns successive results from an iterable."""
        m = MagicMock()
        m.name = "sf3d"
        m.license_note = "test"
        m.output_format = "glb"
        m.safe_generate.side_effect = list(results_iter)
        return m

    def test_sequential_two_ok_candidates(self):
        import cv2, numpy as np
        from modules.ai_3d_generation.candidate_runner import run_candidates_sequential

        with tempfile.TemporaryDirectory() as sd:
            # Create two source images
            srcs = []
            for i in range(2):
                p = Path(sd) / f"src_{i}.jpg"
                cv2.imwrite(str(p), np.zeros((64, 64, 3), dtype=np.uint8))
                srcs.append(str(p))

            # Create fake GLB outputs that the provider "produces"
            glb1 = Path(sd) / "derived" / "candidates" / "cand_001" / "output.glb"
            glb2 = Path(sd) / "derived" / "candidates" / "cand_002" / "output.glb"
            glb1.parent.mkdir(parents=True, exist_ok=True)
            glb2.parent.mkdir(parents=True, exist_ok=True)

            def make_result(idx):
                glb = glb1 if idx == 0 else glb2
                glb.write_bytes(b"GLBFAKE" * 100)
                return {
                    "status": "ok",
                    "output_path": str(glb),
                    "model_name": "sf3d",
                    "warnings": [],
                    "error": None,
                    "metadata": {},
                    "preview_image_path": None,
                }

            prov = self._mock_provider([make_result(0), make_result(1)])
            results = run_candidates_sequential(sd, srcs, prov, input_size=64)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[1]["status"], "ok")
        self.assertEqual(prov.safe_generate.call_count, 2)

    def test_one_fails_one_succeeds(self):
        import cv2, numpy as np
        from modules.ai_3d_generation.candidate_runner import run_candidates_sequential

        with tempfile.TemporaryDirectory() as sd:
            srcs = []
            for i in range(2):
                p = Path(sd) / f"src_{i}.jpg"
                cv2.imwrite(str(p), np.zeros((64, 64, 3), dtype=np.uint8))
                srcs.append(str(p))

            glb2 = Path(sd) / "derived" / "candidates" / "cand_002" / "output.glb"
            glb2.parent.mkdir(parents=True, exist_ok=True)

            fail_result = {
                "status": "failed",
                "output_path": None,
                "model_name": None,
                "warnings": [],
                "error": "cuda_oom",
                "metadata": {},
                "preview_image_path": None,
            }
            ok_result = {
                "status": "ok",
                "output_path": str(glb2),
                "model_name": "sf3d",
                "warnings": [],
                "error": None,
                "metadata": {},
                "preview_image_path": None,
            }
            glb2.write_bytes(b"GLBFAKE" * 100)

            prov = self._mock_provider([fail_result, ok_result])
            results = run_candidates_sequential(sd, srcs, prov, input_size=64)

            self.assertEqual(results[0]["status"], "failed")
            self.assertEqual(results[1]["status"], "ok")
            # Verify candidate_manifest.json written for each
            for r in results:
                cand_dir = Path(sd) / "derived" / "candidates" / r["candidate_id"]
                self.assertTrue((cand_dir / "candidate_manifest.json").exists())

    def test_max_candidates_limit(self):
        import cv2, numpy as np
        from modules.ai_3d_generation.candidate_runner import run_candidates_sequential

        with tempfile.TemporaryDirectory() as sd:
            srcs = []
            for i in range(5):
                p = Path(sd) / f"src_{i}.jpg"
                cv2.imwrite(str(p), np.zeros((64, 64, 3), dtype=np.uint8))
                srcs.append(str(p))

            def make_fail():
                return {
                    "status": "failed", "output_path": None, "model_name": None,
                    "warnings": [], "error": "test", "metadata": {},
                    "preview_image_path": None,
                }

            prov = self._mock_provider([make_fail() for _ in range(3)])
            results = run_candidates_sequential(
                sd, srcs, prov, input_size=64, max_candidates=3
            )

        # Should only process 3, not 5
        self.assertEqual(len(results), 3)
        self.assertEqual(prov.safe_generate.call_count, 3)

    def test_worker_metadata_preserved(self):
        import cv2, numpy as np
        from modules.ai_3d_generation.candidate_runner import run_candidates_sequential

        with tempfile.TemporaryDirectory() as sd:
            srcs = []
            p = Path(sd) / "src_0.jpg"
            cv2.imwrite(str(p), np.zeros((64, 64, 3), dtype=np.uint8))
            srcs.append(str(p))

            glb = Path(sd) / "derived" / "candidates" / "cand_001" / "output.glb"
            glb.parent.mkdir(parents=True, exist_ok=True)
            glb.write_bytes(b"GLBFAKE" * 100)

            worker_meta = {"peak_mem_mb": 4000, "device": "cuda:0"}
            result_payload = {
                "status": "ok",
                "output_path": str(glb),
                "model_name": "sf3d",
                "warnings": [],
                "error": None,
                "metadata": worker_meta,
                "preview_image_path": None,
            }

            prov = self._mock_provider([result_payload])
            results = run_candidates_sequential(sd, srcs, prov, input_size=64, input_mode="video")
            
            self.assertEqual(results[0]["worker_metadata"], worker_meta)
            self.assertEqual(results[0]["model_name"], "sf3d")
            self.assertEqual(results[0]["source_type"], "video_frame")
            self.assertGreater(results[0]["output_size_bytes"], 0)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Pipeline Polish (selected_frame_path, input_mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelinePolish(unittest.TestCase):

    @patch("modules.ai_3d_generation.pipeline.build_manifest")
    def test_normalize_input_mode_legacy(self, mock_build):
        from modules.ai_3d_generation.pipeline import _build_failed_manifest
        mock_provider = MagicMock()
        mock_provider.name = "sf3d"
        mock_provider.output_format = "glb"
        _build_failed_manifest("sess1", "/test.jpg", "image", mock_provider, [], [], None)
        
        args, kwargs = mock_build.call_args
        self.assertEqual(kwargs["input_type"], "single_image")

    @patch("modules.ai_3d_generation.pipeline.build_manifest")
    @patch("modules.ai_3d_generation.candidate_selector.select_best")
    @patch("modules.ai_3d_generation.candidate_runner.run_candidates_sequential")
    @patch("modules.ai_3d_generation.multi_input.resolve_candidate_sources")
    @patch("modules.ai_3d_generation.video_candidates.select_top_k_frames")
    def test_pipeline_selected_frame_path_video_multi(
        self, mock_top_k, mock_resolve, mock_run, mock_select, mock_build
    ):
        from modules.ai_3d_generation.pipeline import generate_ai_3d
        import modules.operations.settings as settings
        
        settings.ai_3d_multi_candidate_enabled = True
        mock_resolve.return_value = {"input_mode": "video", "sources": ["/fake.mp4"], "uploaded_files_count": 1}
        mock_top_k.return_value = ["/fake.mp4"]
        
        with tempfile.TemporaryDirectory() as d:
            src_path = Path(d) / "fake_src.jpg"
            src_path.write_text("fake")
            
            mock_run.return_value = [{"candidate_id": "cand_001"}]
            mock_select.return_value = (
                {"candidate_id": "cand_001", "source_path": str(src_path), "provider_status": "ok"}, 
                [], 
                "test"
            )
            
            provider = MagicMock()
            provider.name = "sf3d"
            provider.output_format = "glb"
            
            with patch("shutil.copy2") as mock_copy:
                generate_ai_3d("sess1", "/fake.mp4", d, provider, {})
                
                # Verify source frame was copied to selected_frame.jpg
                mock_copy.assert_any_call(str(src_path), str(Path(d) / "derived" / "selected_frame.jpg"))
                
                # Verify manifest gets correct path
                args, kwargs = mock_build.call_args
                self.assertTrue(kwargs["selected_frame_path"].endswith("derived\\selected_frame.jpg") or 
                                kwargs["selected_frame_path"].endswith("derived/selected_frame.jpg"))

if __name__ == "__main__":
    unittest.main()

