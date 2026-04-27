"""
Phase 6.1–6.2 Tests: SAM2 + Depth Anything scaffold + Coverage Policy
======================================================================

⚠️  These tests do NOT require torch, SAM2, or Depth Anything.
"""

import pytest
from unittest.mock import patch, MagicMock
from modules.capture_workflow.object_masker import ObjectMasker
from modules.operations.settings import settings
import numpy as np


# ===================================================================
# SAM2 Tests
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
        with patch(
            "modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend"
        ) as mock:
            backend = MagicMock()
            backend.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {"mask_confidence": 0.5})
            mock.return_value = backend
            _, meta = masker.generate_mask(frame)
            called = [c.args[0] for c in mock.call_args_list]
            assert "sam2" not in called
            assert meta.get("fallback_used") is not True


class TestSam2RequestedButDisabled:
    def test_fallback_used_when_sam2_disabled(self):
        with patch("modules.operations.settings.settings.segmentation_method", "sam2"), \
             patch("modules.operations.settings.settings.sam2_enabled", False):
            masker = ObjectMasker()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock:
                backend = MagicMock()
                backend.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {"mask_confidence": 0.5})
                mock.return_value = backend
                _, meta = masker.generate_mask(frame)
                called = [c.args[0] for c in mock.call_args_list]
                assert "sam2" not in called
                assert meta.get("fallback_used") is True
                assert meta.get("requested_segmentation_method") == "sam2"
                assert "SAM2 disabled" in meta.get("fallback_reason", "")
                assert meta.get("segmentation_method") != "sam2"


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
                backend = MagicMock()
                backend.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {"mask_confidence": 0.5})
                mock.return_value = backend
                _, meta = masker.generate_mask(frame)
                assert "sam2" not in [c.args[0] for c in mock.call_args_list]
                assert meta.get("fallback_used") is True
                assert "Checkpoint not found" in meta.get("fallback_reason", "")

    def test_fallback_when_torch_not_installed(self):
        with patch("modules.operations.settings.settings.segmentation_method", "sam2"), \
             patch("modules.operations.settings.settings.sam2_enabled", True):
            masker = ObjectMasker()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock:
                backend = MagicMock()
                backend.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {"mask_confidence": 0.5})
                mock.return_value = backend
                _, meta = masker.generate_mask(frame)
                assert meta.get("fallback_used") is True
                reason = meta.get("fallback_reason", "")
                assert "not" in reason.lower() or "unavailable" in reason.lower()


class TestSam2WrapperStatus:
    def test_status_checkpoint_missing(self):
        with patch("modules.operations.settings.settings.sam2_enabled", True), \
             patch("modules.operations.settings.settings.sam2_checkpoint", "nope.pt"), \
             patch("modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True), \
             patch("modules.ai_segmentation.sam2_wrapper.SAM2_IMPORT_ERROR_REASON", None):
            from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper
            wrapper = SAM2Wrapper()
            status = wrapper.get_status()
            assert status["checkpoint_exists"] is False
            assert status["sam2_available"] is False
            assert "Checkpoint not found" in (status["sam2_error_reason"] or "")

    def test_status_sam2_disabled(self):
        with patch("modules.operations.settings.settings.sam2_enabled", False):
            from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper
            wrapper = SAM2Wrapper()
            status = wrapper.get_status()
            assert status["sam2_enabled"] is False
            assert status["sam2_available"] is False
            assert status["sam2_model_loaded"] is False
            assert status["sam2_inference_ran"] is False

    def test_status_has_all_required_fields(self):
        from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper
        wrapper = SAM2Wrapper()
        status = wrapper.get_status()
        required = {"sam2_enabled", "sam2_available", "sam2_model_loaded",
                     "sam2_inference_ran", "sam2_error_reason", "device",
                     "checkpoint_exists", "model_cfg", "checkpoint"}
        assert required.issubset(status.keys())


class TestSam2BackendFallback:
    def test_not_implemented_triggers_fallback(self):
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
                sam2_be.segment.side_effect = RuntimeError("SAM2 not available")
                mock.return_value = sam2_be
                with pytest.raises(RuntimeError):
                    masker.generate_mask(frame)


class TestSam2ReviewOnly:
    def test_sam2_review_only_default(self):
        assert settings.sam2_review_only is True

    def test_sam2_used_asset_metadata(self):
        mock_predictor = MagicMock()
        with patch("modules.operations.settings.settings.segmentation_method", "sam2"), \
             patch("modules.operations.settings.settings.sam2_enabled", True), \
             patch("modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True), \
             patch("modules.ai_segmentation.sam2_wrapper.SAM2_IMPORT_ERROR_REASON", None), \
             patch("modules.ai_segmentation.sam2_wrapper.build_sam2_video_predictor", create=True, return_value=mock_predictor), \
             patch("pathlib.Path.exists", return_value=True):
            masker = ObjectMasker()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock:
                be = MagicMock()
                be.segment.return_value = (
                    np.zeros((100, 100), dtype=np.uint8),
                    {"mask_confidence": 0.9, "backend_name": "sam2", "segmentation_method": "sam2"},
                )
                mock.return_value = be
                _, meta = masker.generate_mask(frame)
                assert settings.sam2_review_only is True
                assert meta.get("segmentation_method") == "sam2"


# ===================================================================
# Mask Naming Compatibility
# ===================================================================

class TestMaskNamingCompatibility:
    def test_mask_naming_convention(self, tmp_path):
        """Mask file must be named frame_XXXX.jpg.png for COLMAP/OpenMVS."""
        import cv2
        frames_dir = tmp_path / "frames"
        masks_dir = tmp_path / "masks"
        frames_dir.mkdir()

        # Create a dummy frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame, (30, 30), (70, 70), (255, 255, 255), -1)
        cv2.imwrite(str(frames_dir / "frame_0001.jpg"), frame)

        masker = ObjectMasker()
        results = masker.process_session(str(frames_dir), str(masks_dir))

        assert len(results) == 1
        mask_path = results[0]["mask_path"]
        # Convention: frame_0001.jpg → frame_0001.jpg.png
        assert mask_path.endswith("frame_0001.jpg.png")
        assert (masks_dir / "frame_0001.jpg.png").exists()

    def test_mask_naming_png_input(self, tmp_path):
        """PNG input: frame_0001.png → frame_0001.png.png"""
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
        result = generate_prompts((100, 200), mode="center_point")
        assert result["prompt_mode"] == "center_point"
        assert result["prompt_source"] == "frame_center"
        assert result["points"] == [[100, 50]]  # w//2, h//2
        assert result["bbox"] is None

    def test_center_box_no_legacy(self):
        from modules.ai_segmentation.prompting import generate_prompts
        result = generate_prompts((100, 200), mode="center_box")
        assert result["prompt_mode"] == "center_box"
        assert result["prompt_source"] == "frame_center"
        assert result["bbox"] is not None

    def test_auto_with_good_legacy(self):
        from modules.ai_segmentation.prompting import generate_prompts
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[30:70, 50:150] = 255
        meta = {"mask_confidence": 0.8}
        result = generate_prompts((100, 200), mode="auto", legacy_mask=mask, legacy_meta=meta)
        assert result["prompt_mode"] == "center_box"
        assert result["prompt_source"] == "legacy_mask"

    def test_auto_with_low_confidence_legacy(self):
        from modules.ai_segmentation.prompting import generate_prompts
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[30:70, 50:150] = 255
        meta = {"mask_confidence": 0.1}
        result = generate_prompts((100, 200), mode="auto", legacy_mask=mask, legacy_meta=meta)
        assert result["prompt_mode"] == "center_point"

    def test_prompt_metadata_fields(self):
        from modules.ai_segmentation.prompting import generate_prompts
        result = generate_prompts((100, 100), mode="center_point")
        for key in ["prompt_mode", "prompt_source", "bbox", "points", "labels", "confidence"]:
            assert key in result


# ===================================================================
# Depth Anything Scaffold Tests
# ===================================================================

class TestDepthAnythingScaffold:
    def test_depth_anything_disabled_by_default(self):
        assert settings.depth_anything_enabled is False

    def test_depth_anything_review_only_default(self):
        assert settings.depth_anything_review_only is True

    def test_depth_disabled_prevents_imports(self):
        from modules.ai_depth.depth_anything_wrapper import HAS_DEPTH_ANYTHING
        # With default settings (disabled), no heavy imports happen
        assert HAS_DEPTH_ANYTHING is False

    def test_depth_wrapper_status_disabled(self):
        from modules.ai_depth.depth_anything_wrapper import DepthAnythingWrapper
        wrapper = DepthAnythingWrapper()
        status = wrapper.get_status()
        assert status["depth_enabled"] is False
        assert status["depth_available"] is False
        assert status["depth_model_loaded"] is False
        assert "disabled" in (status["depth_error_reason"] or "").lower()

    def test_depth_wrapper_status_all_fields(self):
        from modules.ai_depth.depth_anything_wrapper import DepthAnythingWrapper
        wrapper = DepthAnythingWrapper()
        status = wrapper.get_status()
        required = {"depth_enabled", "depth_available", "depth_model_loaded",
                     "depth_inference_ran", "depth_error_reason", "device",
                     "checkpoint_exists", "model_name", "checkpoint"}
        assert required.issubset(status.keys())

    def test_depth_checkpoint_missing(self):
        with patch("modules.operations.settings.settings.depth_anything_enabled", True), \
             patch("modules.operations.settings.settings.depth_anything_checkpoint", "nope.pth"), \
             patch("modules.ai_depth.depth_anything_wrapper.HAS_DEPTH_ANYTHING", True), \
             patch("modules.ai_depth.depth_anything_wrapper.DEPTH_IMPORT_ERROR_REASON", None):
            from modules.ai_depth.depth_anything_wrapper import DepthAnythingWrapper
            wrapper = DepthAnythingWrapper()
            assert wrapper.checkpoint_exists is False
            assert wrapper.depth_available is False


# ===================================================================
# Depth Prior Policy Tests
# ===================================================================

class TestDepthPriorPolicy:
    def test_rejects_low_iou(self):
        from modules.ai_depth.depth_prior_policy import evaluate_depth_prior_eligibility
        result = evaluate_depth_prior_eligibility(
            segmentation_iou=0.60, leakage_ratio=0.02, mask_confidence=0.80
        )
        assert result["depth_prior_allowed"] is False
        assert "IoU" in result["reason"]

    def test_rejects_high_leakage(self):
        from modules.ai_depth.depth_prior_policy import evaluate_depth_prior_eligibility
        result = evaluate_depth_prior_eligibility(
            segmentation_iou=0.90, leakage_ratio=0.10, mask_confidence=0.80
        )
        assert result["depth_prior_allowed"] is False
        assert "leakage" in result["reason"]

    def test_rejects_low_confidence(self):
        from modules.ai_depth.depth_prior_policy import evaluate_depth_prior_eligibility
        result = evaluate_depth_prior_eligibility(
            segmentation_iou=0.90, leakage_ratio=0.02, mask_confidence=0.50
        )
        assert result["depth_prior_allowed"] is False
        assert "confidence" in result["reason"]

    def test_rejects_when_disabled(self):
        """Even with good metrics, rejected when DEPTH_ANYTHING_ENABLED=false."""
        from modules.ai_depth.depth_prior_policy import evaluate_depth_prior_eligibility
        result = evaluate_depth_prior_eligibility(
            segmentation_iou=0.95, leakage_ratio=0.01, mask_confidence=0.90
        )
        # Default is disabled, so should be rejected
        assert result["depth_prior_allowed"] is False
        assert "DEPTH_ANYTHING_ENABLED=false" in result["reason"]

    def test_allows_when_all_conditions_met(self):
        from modules.ai_depth.depth_prior_policy import evaluate_depth_prior_eligibility
        with patch("modules.operations.settings.settings.depth_anything_enabled", True):
            result = evaluate_depth_prior_eligibility(
                segmentation_iou=0.92, leakage_ratio=0.02, mask_confidence=0.85
            )
            assert result["depth_prior_allowed"] is True
            assert "sufficient" in result["reason"].lower()


# ===================================================================
# Coverage / Completion Policy Tests
# ===================================================================

class TestCoveragePolicy:
    def test_production_candidate(self):
        from modules.ai_depth.depth_prior_policy import classify_coverage
        result = classify_coverage(0.75)
        assert result["status"] == "production_candidate"
        assert result["ai_completion_allowed"] is False

    def test_review_ready(self):
        from modules.ai_depth.depth_prior_policy import classify_coverage
        result = classify_coverage(0.60)
        assert result["status"] == "review_ready"

    def test_preview_only(self):
        from modules.ai_depth.depth_prior_policy import classify_coverage
        result = classify_coverage(0.40)
        assert result["status"] == "preview_only"

    def test_failed_low_coverage(self):
        from modules.ai_depth.depth_prior_policy import classify_coverage
        result = classify_coverage(0.20)
        assert result["status"] == "failed"
        assert result["ai_completion_allowed"] is False

    def test_boundary_70_percent(self):
        from modules.ai_depth.depth_prior_policy import classify_coverage
        result = classify_coverage(0.70)
        assert result["status"] == "production_candidate"

    def test_boundary_50_percent(self):
        from modules.ai_depth.depth_prior_policy import classify_coverage
        result = classify_coverage(0.50)
        assert result["status"] == "review_ready"

    def test_boundary_30_percent(self):
        from modules.ai_depth.depth_prior_policy import classify_coverage
        result = classify_coverage(0.30)
        assert result["status"] == "preview_only"
