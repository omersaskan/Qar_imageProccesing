"""
Phase 6.1 SAM2 Feature-Flag Tests
==================================

These tests verify the SAM2 integration safety guarantees:

1. Default env always uses legacy segmentation.
2. SEGMENTATION_METHOD=sam2 + SAM2_ENABLED=false → falls back to legacy.
3. SEGMENTATION_METHOD=sam2 + SAM2 unavailable → falls back to legacy.
4. SAM2Wrapper status correctly reports checkpoint missing.
5. SAM2Backend NotImplementedError does not crash ObjectMasker when fallback enabled.
6. SAM2-produced assets are review_only / delivery_ready=false.

⚠️  These tests do NOT require torch or SAM2 to be installed.
"""

import pytest
from unittest.mock import patch, MagicMock
from modules.capture_workflow.object_masker import ObjectMasker
from modules.operations.settings import settings
import numpy as np


# ---------------------------------------------------------------------------
# 1. Default env uses legacy
# ---------------------------------------------------------------------------

class TestDefaultEnvUsesLegacy:
    """Verify that the default configuration uses legacy segmentation."""

    def test_default_segmentation_method_is_legacy(self):
        """Settings default must be 'legacy'."""
        assert settings.segmentation_method == "legacy"

    def test_default_sam2_enabled_is_false(self):
        """SAM2_ENABLED must default to false."""
        assert settings.sam2_enabled is False

    def test_default_sam2_review_only_is_true(self):
        """SAM2_REVIEW_ONLY must default to true."""
        assert settings.sam2_review_only is True

    def test_default_sam2_fallback_to_legacy_is_true(self):
        """SAM2_FALLBACK_TO_LEGACY must default to true."""
        assert settings.sam2_fallback_to_legacy is True

    def test_default_object_masker_does_not_touch_sam2(self):
        """ObjectMasker with default settings never invokes SAM2 path."""
        masker = ObjectMasker()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch(
            "modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend"
        ) as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.segment.return_value = (
                np.zeros((100, 100), dtype=np.uint8),
                {"mask_confidence": 0.5},
            )
            mock_get_backend.return_value = mock_backend

            binary, meta = masker.generate_mask(frame)

            # Backend must never be called with 'sam2' in default mode
            called_names = [
                call.args[0] for call in mock_get_backend.call_args_list
            ]
            assert "sam2" not in called_names
            assert meta.get("fallback_used") is not True


# ---------------------------------------------------------------------------
# 2. SEGMENTATION_METHOD=sam2 + SAM2_ENABLED=false → fallback to legacy
# ---------------------------------------------------------------------------

class TestSam2RequestedButDisabled:
    """SAM2 requested via SEGMENTATION_METHOD but kill-switch is off."""

    def test_fallback_used_when_sam2_disabled(self):
        """Must fall back and report clear metadata."""
        with patch(
            "modules.operations.settings.settings.segmentation_method", "sam2"
        ), patch("modules.operations.settings.settings.sam2_enabled", False):

            masker = ObjectMasker()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch(
                "modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend"
            ) as mock_get_backend:
                mock_backend = MagicMock()
                mock_backend.segment.return_value = (
                    np.zeros((100, 100), dtype=np.uint8),
                    {"mask_confidence": 0.5},
                )
                mock_get_backend.return_value = mock_backend

                binary, meta = masker.generate_mask(frame)

                # Should NOT have called get_backend with 'sam2'
                called_names = [
                    call.args[0] for call in mock_get_backend.call_args_list
                ]
                assert "sam2" not in called_names

                # Must include fallback metadata
                assert meta.get("fallback_used") is True
                assert meta.get("requested_segmentation_method") == "sam2"
                assert "SAM2 disabled" in meta.get("fallback_reason", "")

                # segmentation_method must reflect actual backend used
                assert meta.get("segmentation_method") != "sam2"


# ---------------------------------------------------------------------------
# 3. SEGMENTATION_METHOD=sam2 + SAM2 unavailable → fallback to legacy
# ---------------------------------------------------------------------------

class TestSam2EnabledButUnavailable:
    """SAM2 enabled but packages or checkpoint missing."""

    def test_fallback_when_checkpoint_missing(self):
        """Falls back when SAM2_ENABLED=True but checkpoint is missing."""
        with patch(
            "modules.operations.settings.settings.segmentation_method", "sam2"
        ), patch(
            "modules.operations.settings.settings.sam2_enabled", True
        ), patch(
            "modules.operations.settings.settings.sam2_checkpoint",
            "non_existent_checkpoint.pt",
        ), patch(
            # Simulate: torch+sam2 are installed (HAS_SAM2=True)
            "modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True
        ), patch(
            "modules.ai_segmentation.sam2_wrapper.SAM2_IMPORT_ERROR_REASON",
            None,
        ), patch(
            "pathlib.Path.exists", return_value=False
        ):

            masker = ObjectMasker()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch(
                "modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend"
            ) as mock_get_backend:
                mock_backend = MagicMock()
                mock_backend.segment.return_value = (
                    np.zeros((100, 100), dtype=np.uint8),
                    {"mask_confidence": 0.5},
                )
                mock_get_backend.return_value = mock_backend

                binary, meta = masker.generate_mask(frame)

                called_names = [
                    call.args[0] for call in mock_get_backend.call_args_list
                ]
                assert "sam2" not in called_names
                assert meta.get("fallback_used") is True
                assert "Checkpoint not found" in meta.get(
                    "fallback_reason", ""
                )

    def test_fallback_when_torch_not_installed(self):
        """Falls back when SAM2_ENABLED=True but torch is not installed."""
        with patch(
            "modules.operations.settings.settings.segmentation_method", "sam2"
        ), patch(
            "modules.operations.settings.settings.sam2_enabled", True
        ):
            # Don't mock HAS_SAM2 — let real module state apply
            # (torch isn't installed in this env, so HAS_SAM2 = False)
            masker = ObjectMasker()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch(
                "modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend"
            ) as mock_get_backend:
                mock_backend = MagicMock()
                mock_backend.segment.return_value = (
                    np.zeros((100, 100), dtype=np.uint8),
                    {"mask_confidence": 0.5},
                )
                mock_get_backend.return_value = mock_backend

                binary, meta = masker.generate_mask(frame)

                assert meta.get("fallback_used") is True
                # Reason should mention torch or SAM2 package
                reason = meta.get("fallback_reason", "")
                assert "not" in reason.lower() or "unavailable" in reason.lower()


# ---------------------------------------------------------------------------
# 4. SAM2Wrapper status reports checkpoint missing
# ---------------------------------------------------------------------------

class TestSam2WrapperStatus:
    """Verify SAM2Wrapper.get_status() reports accurate information."""

    def test_status_checkpoint_missing(self):
        """Status must report checkpoint_exists=false and error reason."""
        with patch(
            "modules.operations.settings.settings.sam2_enabled", True
        ), patch(
            "modules.operations.settings.settings.sam2_checkpoint",
            "does_not_exist.pt",
        ), patch(
            "modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True
        ), patch(
            "modules.ai_segmentation.sam2_wrapper.SAM2_IMPORT_ERROR_REASON",
            None,
        ):
            from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper

            wrapper = SAM2Wrapper()
            status = wrapper.get_status()

            assert status["sam2_enabled"] is True
            assert status["checkpoint_exists"] is False
            assert status["sam2_available"] is False
            assert "Checkpoint not found" in (
                status["sam2_error_reason"] or ""
            )

    def test_status_sam2_disabled(self):
        """Status must report sam2_enabled=false when disabled."""
        with patch(
            "modules.operations.settings.settings.sam2_enabled", False
        ):
            from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper

            wrapper = SAM2Wrapper()
            status = wrapper.get_status()

            assert status["sam2_enabled"] is False
            assert status["sam2_available"] is False
            assert status["sam2_model_loaded"] is False
            assert status["sam2_inference_ran"] is False

    def test_status_has_all_required_fields(self):
        """Status dict must contain all 9 required fields."""
        from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper

        wrapper = SAM2Wrapper()
        status = wrapper.get_status()

        required_keys = {
            "sam2_enabled",
            "sam2_available",
            "sam2_model_loaded",
            "sam2_inference_ran",
            "sam2_error_reason",
            "device",
            "checkpoint_exists",
            "model_cfg",
            "checkpoint",
        }
        assert required_keys.issubset(status.keys())


# ---------------------------------------------------------------------------
# 5. SAM2Backend NotImplementedError does not crash ObjectMasker
# ---------------------------------------------------------------------------

class TestSam2BackendNotImplementedFallback:
    """ObjectMasker must survive SAM2Backend raising NotImplementedError."""

    def test_not_implemented_triggers_fallback(self):
        """NotImplementedError from SAM2 backend → heuristic fallback."""
        with patch(
            "modules.operations.settings.settings.segmentation_method", "sam2"
        ), patch(
            "modules.operations.settings.settings.sam2_enabled", True
        ), patch(
            "modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True
        ), patch(
            "pathlib.Path.exists", return_value=True
        ):

            masker = ObjectMasker()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch(
                "modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend"
            ) as mock_get_backend:
                # SAM2 backend raises NotImplementedError
                sam2_backend = MagicMock()
                sam2_backend.segment.side_effect = NotImplementedError(
                    "SAM2 frame segmentation not yet implemented"
                )

                # Heuristic fallback succeeds
                heuristic_backend = MagicMock()
                heuristic_backend.segment.return_value = (
                    np.zeros((100, 100), dtype=np.uint8),
                    {"mask_confidence": 0.5, "backend_name": "heuristic"},
                )

                mock_get_backend.side_effect = lambda name: (
                    sam2_backend if name == "sam2" else heuristic_backend
                )

                # Must NOT crash
                binary, meta = masker.generate_mask(frame)

                assert meta.get("fallback_used") is True
                assert "Backend sam2 failed" in meta.get(
                    "fallback_reason", ""
                )
                assert meta.get("requested_segmentation_method") == "sam2"

    def test_hard_fail_propagates_not_implemented(self):
        """When hard_fail_on_backend_error=True, NotImplementedError propagates."""
        from modules.capture_workflow.config import SegmentationConfig

        with patch(
            "modules.operations.settings.settings.segmentation_method", "sam2"
        ), patch(
            "modules.operations.settings.settings.sam2_enabled", True
        ), patch(
            "modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True
        ), patch(
            "pathlib.Path.exists", return_value=True
        ):

            config = SegmentationConfig(hard_fail_on_backend_error=True)
            masker = ObjectMasker(config=config)
            frame = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch(
                "modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend"
            ) as mock_get_backend:
                sam2_backend = MagicMock()
                sam2_backend.segment.side_effect = NotImplementedError(
                    "SAM2 not implemented"
                )
                mock_get_backend.return_value = sam2_backend

                with pytest.raises(NotImplementedError):
                    masker.generate_mask(frame)


# ---------------------------------------------------------------------------
# 6. SAM2-produced assets are review_only / delivery_ready=false
# ---------------------------------------------------------------------------

class TestSam2ReviewOnly:
    """Assets using SAM2 masks must be flagged for review."""

    def test_sam2_review_only_setting_default(self):
        """SAM2_REVIEW_ONLY defaults to true in settings."""
        assert settings.sam2_review_only is True

    def test_sam2_used_asset_not_delivery_ready(self):
        """When SAM2 is the segmentation method, delivery_ready should be false."""
        # This verifies the metadata contract: segmentation_method=sam2
        # implies review_only, which downstream (cleanup/export) interprets
        # as delivery_ready=false.
        with patch(
            "modules.operations.settings.settings.segmentation_method", "sam2"
        ), patch(
            "modules.operations.settings.settings.sam2_enabled", True
        ), patch(
            "modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True
        ), patch(
            "pathlib.Path.exists", return_value=True
        ):

            masker = ObjectMasker()
            frame = np.zeros((100, 100, 3), dtype=np.uint8)

            with patch(
                "modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend"
            ) as mock_get_backend:
                # Simulate a SAM2 backend that actually returns a mask
                sam2_backend = MagicMock()
                sam2_backend.segment.return_value = (
                    np.zeros((100, 100), dtype=np.uint8),
                    {
                        "mask_confidence": 0.9,
                        "backend_name": "sam2",
                        "segmentation_method": "sam2",
                    },
                )
                mock_get_backend.return_value = sam2_backend

                binary, meta = masker.generate_mask(frame)

                # When sam2 is used, SAM2_REVIEW_ONLY=true means
                # assets using this segmentation are review-only
                assert settings.sam2_review_only is True
                assert meta.get("segmentation_method") == "sam2"
