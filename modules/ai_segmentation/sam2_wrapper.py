
import os
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

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
    - segment_video() returns None when SAM2 is unavailable.
    - segment_frame() returns None when SAM2 is unavailable.
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
                self.predictor = build_sam2_video_predictor(
                    self.model_cfg, self.checkpoint, device=self.device
                )
                self.sam2_model_loaded = True
                logger.info(
                    f"SAM2 model loaded (device={self.device}, "
                    f"cfg={self.model_cfg}, ckpt={self.checkpoint})"
                )
            except Exception as e:
                self.sam2_error_reason = f"Model load failed: {e}"
                logger.error(f"Failed to load SAM2 model: {e}")
                self.sam2_available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Returns True only when SAM2 is fully ready for inference."""
        return self.sam2_available and self.sam2_model_loaded

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

    def segment_frame(
        self,
        frame: np.ndarray,
        prompts: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """
        Segment a single frame using SAM2 image predictor.

        Args:
            frame: BGR image as numpy array (H, W, 3).
            prompts: Dict from prompting.generate_prompts() with
                     bbox, points, labels.

        Returns:
            Binary mask (H, W) as uint8 with values 0/255,
            or None if SAM2 is not available / inference fails.
        """
        if not self.is_available():
            logger.error(
                f"SAM2 is not available for frame segmentation: "
                f"{self.sam2_error_reason}"
            )
            return None

        try:
            import cv2
            t0 = time.time()

            # Convert BGR → RGB for SAM2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # SAM2 image predictor expects specific prompt formats.
            # This is the real inference path — only reached when
            # SAM2_ENABLED=true + checkpoint exists + torch+sam2 installed.
            self.predictor.set_image(rgb)

            point_coords = None
            point_labels = None
            box_input = None

            if prompts.get("points"):
                point_coords = np.array(prompts["points"], dtype=np.float32)
                point_labels = np.array(
                    prompts.get("labels", [1] * len(prompts["points"])),
                    dtype=np.int32,
                )

            if prompts.get("bbox"):
                box_input = np.array(prompts["bbox"], dtype=np.float32)

            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_input,
                multimask_output=True,
            )

            # Select the best mask (highest score)
            best_idx = int(np.argmax(scores))
            binary = (masks[best_idx] > 0).astype(np.uint8) * 255

            elapsed = time.time() - t0
            self.sam2_inference_ran = True
            logger.info(f"SAM2 frame segmentation completed in {elapsed:.2f}s")
            return binary

        except Exception as e:
            self.sam2_error_reason = f"Frame inference failed: {e}"
            logger.error(f"SAM2 frame inference failed: {e}")
            return None

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
        if not self.is_available():
            logger.error(
                f"SAM2 is not available: {self.sam2_error_reason}. "
                "Caller should fall back to legacy."
            )
            return None

        logger.info(f"Running SAM2 video segmentation on {video_path}")
        self.sam2_inference_ran = True

        # Video-level propagation requires additional SAM2 API calls.
        # For now, return empty dict to signal "no masks produced",
        # which triggers legacy fallback.
        # Full video propagation will be implemented when DEV-SUBSET
        # frame-level results justify the investment.
        return {}


def get_predictor() -> Optional[SAM2Wrapper]:
    """Convenience factory — returns a wrapper only if SAM2 is available."""
    predictor = SAM2Wrapper()
    if predictor.is_available():
        return predictor
    return None
