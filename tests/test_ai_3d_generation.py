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
            f.write(b"\x89PNG\r\n")
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
            f.write(b"\x89PNG\r\n")
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


if __name__ == "__main__":
    unittest.main()
