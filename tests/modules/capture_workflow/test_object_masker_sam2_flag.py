"""
Phase 6.1–6.2 Tests: SAM2 image-mode + Depth Anything scaffold + Coverage
==========================================================================
⚠️  No torch, SAM2, or Depth Anything required.
"""
import pytest
from unittest.mock import patch, MagicMock
from modules.capture_workflow.object_masker import ObjectMasker
from modules.operations.settings import settings
import numpy as np


# ===================================================================
# SAM2 Default / Disabled Tests
# ===================================================================

class TestDefaultEnvUsesLegacy:
    def test_default_segmentation_method_is_legacy(self):
        assert settings.segmentation_method == "legacy"

    def test_default_sam2_enabled_is_false(self):
        assert settings.sam2_enabled is False

    def test_default_sam2_review_only_is_true(self):
        assert settings.sam2_review_only is True

    def test_default_sam2_fallback_to_legacy_is_true(self):
        assert settings.sam2_fallback_to_legacy is True

    def test_default_object_masker_does_not_touch_sam2(self):
        masker = ObjectMasker()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock:
            be = MagicMock()
            be.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {"mask_confidence": 0.5})
            mock.return_value = be
            _, meta = masker.generate_mask(frame)
            assert "sam2" not in [c.args[0] for c in mock.call_args_list]
            assert meta.get("fallback_used") is not True


class TestSam2RequestedButDisabled:
    def test_fallback_used_when_sam2_disabled(self):
        with patch("modules.operations.settings.settings.segmentation_method", "sam2"), \
             patch("modules.operations.settings.settings.sam2_enabled", False):
            masker = ObjectMasker()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock:
                be = MagicMock()
                be.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {"mask_confidence": 0.5})
                mock.return_value = be
                _, meta = masker.generate_mask(frame)
                assert "sam2" not in [c.args[0] for c in mock.call_args_list]
                assert meta.get("fallback_used") is True
                assert "SAM2 disabled" in meta.get("fallback_reason", "")


class TestSam2EnabledButUnavailable:
    def test_fallback_when_checkpoint_missing(self):
        with patch("modules.operations.settings.settings.segmentation_method", "sam2"), \
             patch("modules.operations.settings.settings.sam2_enabled", True), \
             patch("modules.operations.settings.settings.sam2_checkpoint", "nonexistent.pt"), \
             patch("modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True), \
             patch("modules.ai_segmentation.sam2_wrapper.SAM2_IMPORT_ERROR_REASON", None), \
             patch("pathlib.Path.exists", return_value=False):
            masker = ObjectMasker()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock:
                be = MagicMock()
                be.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {"mask_confidence": 0.5})
                mock.return_value = be
                _, meta = masker.generate_mask(frame)
                assert meta.get("fallback_used") is True
                assert "Checkpoint not found" in meta.get("fallback_reason", "")

    def test_fallback_when_torch_not_installed(self):
        with patch("modules.operations.settings.settings.segmentation_method", "sam2"), \
             patch("modules.operations.settings.settings.sam2_enabled", True):
            masker = ObjectMasker()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock:
                be = MagicMock()
                be.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {"mask_confidence": 0.5})
                mock.return_value = be
                _, meta = masker.generate_mask(frame)
                assert meta.get("fallback_used") is True


# ===================================================================
# SAM2 Wrapper Status
# ===================================================================

class TestSam2WrapperStatus:
    def test_status_checkpoint_missing(self):
        with patch("modules.operations.settings.settings.sam2_enabled", True), \
             patch("modules.operations.settings.settings.sam2_checkpoint", "nope.pt"), \
             patch("modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True), \
             patch("modules.ai_segmentation.sam2_wrapper.SAM2_IMPORT_ERROR_REASON", None):
            from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper
            w = SAM2Wrapper()
            s = w.get_status()
            assert s["checkpoint_exists"] is False
            assert s["sam2_available"] is False
            assert "Checkpoint not found" in (s["sam2_error_reason"] or "")

    def test_status_sam2_disabled(self):
        with patch("modules.operations.settings.settings.sam2_enabled", False):
            from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper
            w = SAM2Wrapper()
            s = w.get_status()
            assert s["sam2_enabled"] is False
            assert s["sam2_model_loaded"] is False

    def test_status_has_all_required_fields(self):
        from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper
        w = SAM2Wrapper()
        s = w.get_status()
        required = {"sam2_enabled", "sam2_available", "sam2_model_loaded",
                     "sam2_inference_ran", "sam2_error_reason", "device",
                     "checkpoint_exists", "model_cfg", "checkpoint",
                     "sam2_mode", "temporal_consistency", "api_type"}
        assert required.issubset(s.keys())

    def test_status_mode_fields_image_mode(self):
        """Image-mode wrapper must report correct mode/api_type."""
        from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper, SAM2_IMAGE_MODE
        w = SAM2Wrapper()
        s = w.get_status()
        assert s["sam2_mode"] == SAM2_IMAGE_MODE
        assert s["temporal_consistency"] is False
        assert s["api_type"] == "image_predictor"


# ===================================================================
# SAM2 Backend Fallback
# ===================================================================

class TestSam2BackendFallback:
    def test_runtime_error_triggers_fallback(self):
        with patch("modules.operations.settings.settings.segmentation_method", "sam2"), \
             patch("modules.operations.settings.settings.sam2_enabled", True), \
             patch("modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True), \
             patch("pathlib.Path.exists", return_value=True):
            masker = ObjectMasker()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock:
                sam2_be = MagicMock()
                sam2_be.segment.side_effect = RuntimeError("SAM2 not available")
                heur_be = MagicMock()
                heur_be.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {"mask_confidence": 0.5})
                mock.side_effect = lambda n: sam2_be if n == "sam2" else heur_be
                _, meta = masker.generate_mask(frame)
                assert meta.get("fallback_used") is True
                assert meta.get("requested_segmentation_method") == "sam2"

    def test_hard_fail_propagates(self):
        from modules.capture_workflow.config import SegmentationConfig
        with patch("modules.operations.settings.settings.segmentation_method", "sam2"), \
             patch("modules.operations.settings.settings.sam2_enabled", True), \
             patch("modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True), \
             patch("pathlib.Path.exists", return_value=True):
            config = SegmentationConfig(hard_fail_on_backend_error=True)
            masker = ObjectMasker(config=config)
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock:
                sam2_be = MagicMock()
                sam2_be.segment.side_effect = RuntimeError("SAM2 failed")
                mock.return_value = sam2_be
                with pytest.raises(RuntimeError):
                    masker.generate_mask(frame)


# ===================================================================
# SAM2 Review Only
# ===================================================================

class TestSam2ReviewOnly:
    def test_sam2_review_only_default(self):
        assert settings.sam2_review_only is True

    def test_sam2_used_asset_metadata(self):
        mock_model = MagicMock()
        mock_img_pred = MagicMock()
        mock_sam2 = MagicMock()
        mock_sam2.build_sam.build_sam2.return_value = mock_model
        mock_sam2.sam2_image_predictor.SAM2ImagePredictor.return_value = mock_img_pred
        
        with patch("modules.operations.settings.settings.segmentation_method", "sam2"), \
             patch("modules.operations.settings.settings.sam2_enabled", True), \
             patch("modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True), \
             patch("modules.ai_segmentation.sam2_wrapper.SAM2_IMPORT_ERROR_REASON", None), \
             patch.dict("sys.modules", {"sam2": mock_sam2, "sam2.build_sam": mock_sam2.build_sam, "sam2.sam2_image_predictor": mock_sam2.sam2_image_predictor}), \
             patch("pathlib.Path.exists", return_value=True):
            masker = ObjectMasker()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock:
                be = MagicMock()
                be.segment.return_value = (
                    np.zeros((100, 100), dtype=np.uint8),
                    {"mask_confidence": 0.9, "segmentation_method": "sam2",
                     "sam2_mode": "image_frame", "temporal_consistency": False,
                     "api_type": "image_predictor"},
                )
                mock.return_value = be
                _, meta = masker.generate_mask(frame)
                assert meta.get("segmentation_method") == "sam2"
                assert meta.get("sam2_mode") == "image_frame"
                assert meta.get("temporal_consistency") is False
                assert meta.get("api_type") == "image_predictor"


# ===================================================================
# SAM2 Image Predictor — Fake Predictor Test
# ===================================================================

class TestSam2FakePredictor:
    """Validates image-mode path with a dependency-free fake predictor."""

    def test_segment_frame_with_fake_predictor(self):
        """Fake predictor implementing set_image/predict validates image mode."""
        from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper

        class FakeImagePredictor:
            """Mimics SAM2ImagePredictor API: set_image + predict."""
            def __init__(self):
                self.image_set = False
            def set_image(self, image):
                self.image_set = True
                assert image.shape[2] == 3  # RGB
            def predict(self, point_coords=None, point_labels=None,
                        box=None, multimask_output=True):
                assert self.image_set, "set_image must be called before predict"
                h, w = 100, 100
                masks = np.array([np.ones((h, w), dtype=np.float32)])
                scores = np.array([0.95])
                logits = np.array([np.zeros((h, w), dtype=np.float32)])
                return masks, scores, logits

        with patch("modules.operations.settings.settings.sam2_enabled", True), \
             patch("modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True), \
             patch("modules.ai_segmentation.sam2_wrapper.SAM2_IMPORT_ERROR_REASON", None):
            wrapper = SAM2Wrapper.__new__(SAM2Wrapper)
            wrapper.sam2_enabled = True
            wrapper.sam2_available = True
            wrapper.sam2_model_loaded = True
            wrapper.sam2_inference_ran = False
            wrapper.sam2_error_reason = None
            wrapper.device = "cpu"
            wrapper.checkpoint = "fake.pt"
            wrapper.model_cfg = "fake.yaml"
            wrapper.checkpoint_exists = True
            wrapper.sam2_mode = "image_frame"
            wrapper.temporal_consistency = False
            wrapper.api_type = "image_predictor"
            wrapper.predictor = FakeImagePredictor()

            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            prompt = {"points": [[50, 50]], "labels": [1], "bbox": None}
            mask = wrapper.segment_frame(frame, prompt)

            assert mask is not None
            assert mask.shape == (100, 100)
            assert mask.dtype == np.uint8
            assert np.max(mask) == 255
            assert wrapper.sam2_inference_ran is True
            assert wrapper.predictor.image_set is True

    def test_video_predictor_api_not_called_in_image_mode(self):
        """Proves video predictor methods are never called in image mode."""
        from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper

        class StrictVideoPredictor:
            """Has video API methods that should NEVER be called."""
            def init_state(self, *a, **kw):
                raise AssertionError("init_state must NOT be called in image mode")
            def add_new_points_or_box(self, *a, **kw):
                raise AssertionError("add_new_points_or_box must NOT be called")
            def propagate_in_video(self, *a, **kw):
                raise AssertionError("propagate_in_video must NOT be called")
            # Image API — this IS expected
            def set_image(self, image):
                pass
            def predict(self, **kw):
                h, w = 100, 100
                return (np.array([np.ones((h, w))]),
                        np.array([0.9]),
                        np.array([np.zeros((h, w))]))

        with patch("modules.operations.settings.settings.sam2_enabled", True), \
             patch("modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True), \
             patch("modules.ai_segmentation.sam2_wrapper.SAM2_IMPORT_ERROR_REASON", None):
            wrapper = SAM2Wrapper.__new__(SAM2Wrapper)
            wrapper.sam2_enabled = True
            wrapper.sam2_available = True
            wrapper.sam2_model_loaded = True
            wrapper.sam2_inference_ran = False
            wrapper.sam2_error_reason = None
            wrapper.device = "cpu"
            wrapper.checkpoint = "fake.pt"
            wrapper.model_cfg = "fake.yaml"
            wrapper.checkpoint_exists = True
            wrapper.sam2_mode = "image_frame"
            wrapper.temporal_consistency = False
            wrapper.api_type = "image_predictor"
            wrapper.predictor = StrictVideoPredictor()

            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            prompt = {"points": [[50, 50]], "labels": [1], "bbox": None}
            # This must NOT raise — proves video API is never touched
            mask = wrapper.segment_frame(frame, prompt)
            assert mask is not None


# ===================================================================
# Mask Naming Compatibility
# ===================================================================

class TestMaskNamingCompatibility:
    def test_mask_naming_convention(self, tmp_path):
        import cv2
        frames_dir = tmp_path / "frames"
        masks_dir = tmp_path / "masks"
        frames_dir.mkdir()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame, (30, 30), (70, 70), (255, 255, 255), -1)
        cv2.imwrite(str(frames_dir / "frame_0001.jpg"), frame)
        masker = ObjectMasker()
        results = masker.process_session(str(frames_dir), str(masks_dir))
        assert len(results) == 1
        assert results[0]["mask_path"].endswith("frame_0001.jpg.png")

    def test_mask_naming_png_input(self, tmp_path):
        import cv2
        frames_dir = tmp_path / "frames"
        masks_dir = tmp_path / "masks"
        frames_dir.mkdir()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame, (30, 30), (70, 70), (255, 255, 255), -1)
        cv2.imwrite(str(frames_dir / "frame_0002.png"), frame)
        masker = ObjectMasker()
        results = masker.process_session(str(frames_dir), str(masks_dir))
        assert len(results) == 1
        assert results[0]["mask_path"].endswith("frame_0002.png.png")


# ===================================================================
# Prompting Tests
# ===================================================================

class TestPrompting:
    def test_center_point_no_legacy(self):
        from modules.ai_segmentation.prompting import generate_prompts
        r = generate_prompts((100, 200), mode="center_point")
        assert r["prompt_mode"] == "center_point"
        assert r["prompt_source"] == "frame_center"
        assert r["points"] == [[100, 50]]
        assert r["bbox"] is None

    def test_center_box_no_legacy(self):
        from modules.ai_segmentation.prompting import generate_prompts
        r = generate_prompts((100, 200), mode="center_box")
        assert r["prompt_mode"] == "center_box"
        assert r["bbox"] is not None

    def test_auto_with_good_legacy(self):
        from modules.ai_segmentation.prompting import generate_prompts
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[30:70, 50:150] = 255
        r = generate_prompts((100, 200), mode="auto", legacy_mask=mask,
                             legacy_meta={"mask_confidence": 0.8})
        assert r["prompt_mode"] == "center_box"
        assert r["prompt_source"] == "legacy_mask"

    def test_auto_with_low_confidence(self):
        from modules.ai_segmentation.prompting import generate_prompts
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[30:70, 50:150] = 255
        r = generate_prompts((100, 200), mode="auto", legacy_mask=mask,
                             legacy_meta={"mask_confidence": 0.1})
        assert r["prompt_mode"] == "center_point"

    def test_prompt_metadata_fields(self):
        from modules.ai_segmentation.prompting import generate_prompts
        r = generate_prompts((100, 100), mode="center_point")
        for k in ["prompt_mode", "prompt_source", "bbox", "points", "labels", "confidence"]:
            assert k in r


# ===================================================================
# Depth Anything Scaffold Tests
# ===================================================================

class TestDepthAnythingScaffold:
    def test_depth_disabled_by_default(self):
        assert settings.depth_anything_enabled is False

    def test_depth_review_only_default(self):
        assert settings.depth_anything_review_only is True

    def test_depth_disabled_prevents_imports(self):
        from modules.ai_depth.depth_anything_wrapper import HAS_DEPTH_ANYTHING
        assert HAS_DEPTH_ANYTHING is False

    def test_depth_wrapper_status_disabled(self):
        from modules.ai_depth.depth_anything_wrapper import DepthAnythingWrapper
        s = DepthAnythingWrapper().get_status()
        assert s["depth_enabled"] is False
        assert s["depth_available"] is False
        assert "disabled" in (s["depth_error_reason"] or "").lower()

    def test_depth_wrapper_all_status_fields(self):
        from modules.ai_depth.depth_anything_wrapper import DepthAnythingWrapper
        s = DepthAnythingWrapper().get_status()
        required = {"depth_enabled", "depth_available", "depth_model_loaded",
                     "depth_inference_ran", "depth_error_reason", "device",
                     "checkpoint_exists", "model_name", "checkpoint"}
        assert required.issubset(s.keys())

    def test_depth_checkpoint_missing(self):
        with patch("modules.operations.settings.settings.depth_anything_enabled", True), \
             patch("modules.operations.settings.settings.depth_anything_checkpoint", "nope.pth"), \
             patch("modules.ai_depth.depth_anything_wrapper.HAS_DEPTH_ANYTHING", True), \
             patch("modules.ai_depth.depth_anything_wrapper.DEPTH_IMPORT_ERROR_REASON", None):
            from modules.ai_depth.depth_anything_wrapper import DepthAnythingWrapper
            w = DepthAnythingWrapper()
            assert w.checkpoint_exists is False
            assert w.depth_available is False


# ===================================================================
# Depth Prior Policy Tests
# ===================================================================

class TestDepthPriorPolicy:
    def test_rejects_low_iou(self):
        from modules.ai_depth.depth_prior_policy import evaluate_depth_prior_eligibility
        r = evaluate_depth_prior_eligibility(0.60, 0.02, 0.80)
        assert r["depth_prior_allowed"] is False
        assert "IoU" in r["reason"]

    def test_rejects_high_leakage(self):
        from modules.ai_depth.depth_prior_policy import evaluate_depth_prior_eligibility
        r = evaluate_depth_prior_eligibility(0.90, 0.10, 0.80)
        assert r["depth_prior_allowed"] is False
        assert "leakage" in r["reason"]

    def test_rejects_low_confidence(self):
        from modules.ai_depth.depth_prior_policy import evaluate_depth_prior_eligibility
        r = evaluate_depth_prior_eligibility(0.90, 0.02, 0.50)
        assert r["depth_prior_allowed"] is False

    def test_rejects_when_disabled(self):
        from modules.ai_depth.depth_prior_policy import evaluate_depth_prior_eligibility
        r = evaluate_depth_prior_eligibility(0.95, 0.01, 0.90)
        assert r["depth_prior_allowed"] is False
        assert "DEPTH_ANYTHING_ENABLED=false" in r["reason"]

    def test_allows_when_all_pass(self):
        from modules.ai_depth.depth_prior_policy import evaluate_depth_prior_eligibility
        with patch("modules.operations.settings.settings.depth_anything_enabled", True):
            r = evaluate_depth_prior_eligibility(0.92, 0.02, 0.85)
            assert r["depth_prior_allowed"] is True


# ===================================================================
# Coverage / Completion Policy Tests
# ===================================================================

class TestCoveragePolicy:
    def test_production_candidate(self):
        from modules.ai_depth.depth_prior_policy import classify_coverage
        assert classify_coverage(0.75)["status"] == "production_candidate"

    def test_review_ready(self):
        from modules.ai_depth.depth_prior_policy import classify_coverage
        assert classify_coverage(0.60)["status"] == "review_ready"

    def test_preview_only(self):
        from modules.ai_depth.depth_prior_policy import classify_coverage
        assert classify_coverage(0.40)["status"] == "preview_only"

    def test_failed_low_coverage(self):
        from modules.ai_depth.depth_prior_policy import classify_coverage
        r = classify_coverage(0.20)
        assert r["status"] == "failed"
        assert r["ai_completion_allowed"] is False

    def test_boundary_70(self):
        from modules.ai_depth.depth_prior_policy import classify_coverage
        assert classify_coverage(0.70)["status"] == "production_candidate"

    def test_boundary_50(self):
        from modules.ai_depth.depth_prior_policy import classify_coverage
        assert classify_coverage(0.50)["status"] == "review_ready"

    def test_boundary_30(self):
        from modules.ai_depth.depth_prior_policy import classify_coverage
        assert classify_coverage(0.30)["status"] == "preview_only"
