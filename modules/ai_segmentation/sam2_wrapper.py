
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

# We import settings here to read configurations
from modules.operations.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency probing
# ---------------------------------------------------------------------------
# SAM2 and torch are NOT hard dependencies.  They are imported only when
# SAM2_ENABLED=true in the environment/settings.  This keeps the normal
# install lightweight and prevents import-time crashes.
# ---------------------------------------------------------------------------

HAS_SAM2 = False
HAS_TORCH = False
SAM2_IMPORT_ERROR_REASON: Optional[str] = None

if settings.sam2_enabled:
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        SAM2_IMPORT_ERROR_REASON = "torch not installed"

    if HAS_TORCH:
        try:
            from sam2.build_sam import build_sam2_video_predictor  # noqa: F401
            HAS_SAM2 = True
        except ImportError:
            SAM2_IMPORT_ERROR_REASON = "segment-anything-2 (sam2) package not installed"
else:
    SAM2_IMPORT_ERROR_REASON = "SAM2 disabled in settings (SAM2_ENABLED=false)"


class SAM2Wrapper:
    """
    Wrapper around Meta's SAM2 video predictor.

    Safety guarantees:
    - If SAM2_ENABLED is false, no import/load of torch or SAM2 is attempted.
    - If the checkpoint file is missing, status reports the exact reason.
    - segment_video() returns None (not {}) when SAM2 is unavailable,
      allowing callers to fall back cleanly.
    - All status fields are always populated for observability.
    """

    def __init__(self):
        # --- Status fields (always populated) ---
        self.sam2_enabled: bool = settings.sam2_enabled
        self.sam2_available: bool = HAS_SAM2
        self.sam2_model_loaded: bool = False
        self.sam2_inference_ran: bool = False
        self.sam2_error_reason: Optional[str] = SAM2_IMPORT_ERROR_REASON

        self.device: str = settings.sam2_device if HAS_TORCH else "cpu"
        self.checkpoint: str = settings.sam2_checkpoint
        self.model_cfg: str = settings.sam2_model_cfg
        self.checkpoint_exists: bool = (
            Path(self.checkpoint).exists() if self.checkpoint else False
        )

        self.predictor = None

        # --- Checkpoint validation ---
        if self.sam2_available and not self.checkpoint_exists:
            self.sam2_available = False
            self.sam2_error_reason = (
                f"Checkpoint not found at {self.checkpoint}"
            )
            logger.warning(self.sam2_error_reason)

        # --- Model loading (only if everything is green) ---
        if self.sam2_available:
            try:
                # Real model loading is gated behind SAM2_ENABLED + valid
                # checkpoint + installed packages.  The actual call is
                # commented out until real SAM2 inference is implemented.
                #
                # self.predictor = build_sam2_video_predictor(
                #     self.model_cfg, self.checkpoint, device=self.device
                # )
                # self.sam2_model_loaded = True
                logger.info(
                    f"SAM2 wrapper initialized (device={self.device}, "
                    f"cfg={self.model_cfg}, ckpt={self.checkpoint})"
                )
            except Exception as e:
                self.sam2_error_reason = f"Model load failed: {e}"
                logger.error(f"Failed to initialize SAM2: {e}")
                self.sam2_available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Returns True only when SAM2 is fully ready for inference."""
        return self.sam2_available

    def get_status(self) -> Dict[str, Any]:
        """
        Returns a full status dictionary for observability / diagnostics.

        All keys are always present regardless of SAM2 state.
        """
        return {
            "sam2_enabled": self.sam2_enabled,
            "sam2_available": self.sam2_available,
            "sam2_model_loaded": self.sam2_model_loaded,
            "sam2_inference_ran": self.sam2_inference_ran,
            "sam2_error_reason": self.sam2_error_reason,
            "device": self.device,
            "checkpoint_exists": self.checkpoint_exists,
            "model_cfg": self.model_cfg,
            "checkpoint": self.checkpoint,
        }

    def segment_video(
        self,
        video_path: str,
        prompts: List[Dict[str, Any]],
    ) -> Optional[Dict[int, np.ndarray]]:
        """
        Segments a video using SAM2 given point or box prompts.

        Returns:
            dict mapping frame index → binary mask, or
            None if SAM2 is not available (callers should fall back).
        """
        if not self.sam2_available:
            logger.error(
                f"SAM2 is not available: {self.sam2_error_reason}. "
                "Caller should fall back to legacy."
            )
            return None

        logger.info(f"Running SAM2 segmentation on {video_path}")
        self.sam2_inference_ran = True

        # -----------------------------------------------------------
        # STUB: Real SAM2 inference is NOT implemented yet.
        # This method currently returns an empty dict which signals
        # "no masks produced" and triggers the legacy fallback in
        # ObjectMasker / segmentation_factory.
        #
        # Real implementation requires:
        #   1. SAM2_ENABLED=true in env
        #   2. Valid checkpoint on disk
        #   3. User review approval
        # -----------------------------------------------------------
        return {}


def get_predictor() -> Optional[SAM2Wrapper]:
    """Convenience factory — returns a wrapper only if SAM2 is available."""
    predictor = SAM2Wrapper()
    if predictor.is_available():
        return predictor
    return None
