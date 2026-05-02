"""
SAM2 Wrapper — Image-mode (Option A) for DEV-SUBSET
=====================================================

API design:
- Uses build_sam2() + SAM2ImagePredictor for per-frame segmentation.
- segment_frame() calls predictor.set_image() and predictor.predict().
- segment_video() remains a stub (video propagation is future Option B).

This is explicitly NOT temporal/video propagation.  The video predictor
(build_sam2_video_predictor) uses a completely different API
(init_state / add_new_points_or_box / propagate_in_video) and is NOT
used here.

⚠️  torch and sam2 are NOT hard dependencies.
"""

import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from modules.operations.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------
SAM2_IMAGE_MODE = "image_frame"
SAM2_VIDEO_MODE = "video_temporal"

HAS_SAM2: bool = False  # set to True at runtime if sam2 imports succeed


def probe_sam2_availability() -> Tuple[bool, bool, Optional[str]]:
    """
    Probes for torch and sam2 dependencies based on current settings.
    Returns: (has_sam2, has_torch, error_reason)
    """
    has_sam2 = False
    has_torch = False
    error_reason = None

    if not settings.sam2_enabled:
        return False, False, "SAM2 disabled in settings (SAM2_ENABLED=false)"

    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        return False, False, "torch not installed"

    try:
        # Image-mode: build_sam2 + SAM2ImagePredictor
        from sam2.build_sam import build_sam2  # noqa: F401
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
        has_sam2 = True
    except ImportError:
        error_reason = "segment-anything-2 (sam2) package not installed"

    return has_sam2, has_torch, error_reason


class SAM2Wrapper:
    """
    Wrapper around Meta's SAM2 **image** predictor (Option A).

    API mapping (correct per official SAM2 docs):
    ┌──────────────────────────────────────────────────────────┐
    │ Image mode (this wrapper):                               │
    │   build_sam2() → SAM2ImagePredictor(model)               │
    │   predictor.set_image(rgb)                               │
    │   predictor.predict(point_coords, point_labels, box)     │
    ├──────────────────────────────────────────────────────────┤
    │ Video mode (NOT used here, future Option B):             │
    │   build_sam2_video_predictor()                           │
    │   predictor.init_state(video_path)                       │
    │   predictor.add_new_points_or_box(...)                   │
    │   predictor.propagate_in_video(...)                      │
    └──────────────────────────────────────────────────────────┘

    Safety:
    - If SAM2_ENABLED is false, no import/load of torch or SAM2.
    - If checkpoint is missing, status reports exact reason.
    - segment_frame() returns None when SAM2 is unavailable.
    - All status fields always populated.
    """

    def __init__(self):
        # --- Dynamic Dependency Probing ---
        # We probe at init time to allow runtime settings toggling (e.g. for eval)
        has_sam2, has_torch, error_reason = probe_sam2_availability()

        # --- Status fields ---
        self.sam2_enabled: bool = settings.sam2_enabled
        self.sam2_available: bool = has_sam2
        self.sam2_model_loaded: bool = False
        self.sam2_inference_ran: bool = False
        self.sam2_error_reason: Optional[str] = error_reason

        self.device: str = settings.sam2_device if has_torch else "cpu"
        self.checkpoint: str = settings.sam2_checkpoint
        self.model_cfg: str = settings.sam2_model_cfg
        self.checkpoint_exists: bool = (
            Path(self.checkpoint).exists() if self.checkpoint else False
        )

        # Mode tracking — image mode for DEV-SUBSET
        self.sam2_mode: str = SAM2_IMAGE_MODE
        self.temporal_consistency: bool = False
        self.api_type: str = "image_predictor"

        self.predictor = None

        # --- Checkpoint validation ---
        if self.sam2_available and not self.checkpoint_exists:
            self.sam2_available = False
            self.sam2_error_reason = (
                f"Checkpoint not found at {self.checkpoint}"
            )
            logger.warning(self.sam2_error_reason)

        # --- Model loading (image mode) ---
        if self.sam2_available:
            try:
                # Local imports for builders to prevent global dependency issues
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                model = build_sam2(
                    self.model_cfg,
                    self.checkpoint,
                    device=self.device,
                )
                self.predictor = SAM2ImagePredictor(model)
                self.sam2_model_loaded = True
                logger.info(
                    f"SAM2 image predictor loaded "
                    f"(device={self.device}, cfg={self.model_cfg}, "
                    f"ckpt={self.checkpoint})"
                )
            except Exception as e:
                self.sam2_error_reason = f"Model load failed: {e}"
                logger.error(f"Failed to load SAM2 model: {e}")
                self.sam2_available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """True only when SAM2 image predictor is fully ready."""
        return self.sam2_available and self.sam2_model_loaded

    def get_status(self) -> Dict[str, Any]:
        """Full status dictionary — all keys always present."""
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
            "sam2_mode": self.sam2_mode,
            "temporal_consistency": self.temporal_consistency,
            "api_type": self.api_type,
        }

    def segment_frame(
        self,
        frame: np.ndarray,
        prompts: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """
        Segment a single frame using SAM2ImagePredictor.

        Uses the correct image-mode API:
            predictor.set_image(rgb)
            predictor.predict(point_coords, point_labels, box)

        Args:
            frame: BGR image (H, W, 3).
            prompts: Dict from prompting.generate_prompts().

        Returns:
            Binary mask (H, W) uint8 0/255, or None on failure.
        """
        if not self.is_available():
            logger.error(
                f"SAM2 not available for frame segmentation: "
                f"{self.sam2_error_reason}"
            )
            return None

        try:
            import cv2
            t0 = time.time()

            # Convert BGR → RGB for SAM2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # SAM2ImagePredictor API:
            #   set_image(image) — encodes image features
            #   predict(point_coords, point_labels, box, multimask_output)
            self.predictor.set_image(rgb)

            point_coords = None
            point_labels = None
            box_input = None

            if prompts.get("points"):
                point_coords = np.array(
                    prompts["points"], dtype=np.float32
                )
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

            # Select highest-scoring mask
            best_idx = int(np.argmax(scores))
            binary = (masks[best_idx] > 0).astype(np.uint8) * 255

            elapsed = time.time() - t0
            self.sam2_inference_ran = True
            logger.info(
                f"SAM2 image-mode segmentation: {elapsed:.2f}s "
                f"(score={scores[best_idx]:.3f})"
            )
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
        Video-mode segmentation stub (future Option B).

        This would use:
            build_sam2_video_predictor()
            predictor.init_state(video_path)
            predictor.add_new_points_or_box(...)
            predictor.propagate_in_video(...)

        Currently returns empty dict → triggers legacy fallback.
        """
        if not self.is_available():
            logger.error(
                f"SAM2 not available: {self.sam2_error_reason}. "
                "Caller should fall back to legacy."
            )
            return None

        logger.info(f"SAM2 video-mode not implemented. Path: {video_path}")
        return {}


def get_predictor() -> Optional[SAM2Wrapper]:
    """Convenience factory — returns wrapper only if SAM2 is available."""
    wrapper = SAM2Wrapper()
    if wrapper.is_available():
        return wrapper
    return None
