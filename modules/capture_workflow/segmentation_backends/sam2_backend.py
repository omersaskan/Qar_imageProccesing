
import time
import numpy as np
import logging
from typing import Dict, Any, Tuple
from .base import SegmentationBackend
from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper
from modules.ai_segmentation.prompting import generate_prompts
from modules.operations.settings import settings

logger = logging.getLogger(__name__)


class SAM2Backend(SegmentationBackend):
    """
    Segmentation backend that delegates to the SAM2Wrapper.

    Behavior:
    - If SAM2 is available AND the model is loaded, runs real inference
      via SAM2Wrapper.segment_frame() and returns the mask + metadata.
    - If SAM2 is not available, raises RuntimeError so ObjectMasker
      can catch it and fall back to the heuristic backend.
    - If inference fails at runtime, raises RuntimeError (caught by
      ObjectMasker for fallback).
    """

    def __init__(self):
        self.wrapper = SAM2Wrapper()

    def segment(
        self, frame: np.ndarray, config: Any
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.wrapper.is_available():
            raise RuntimeError(
                f"SAM2 backend requested but SAM2 is not available: "
                f"{self.wrapper.sam2_error_reason}"
            )

        h, w = frame.shape[:2]
        t0 = time.time()

        # Generate prompt using the configured strategy
        prompt = generate_prompts(
            frame_shape=(h, w),
            mode=settings.sam2_prompt_mode,
        )

        # Run real SAM2 inference
        mask = self.wrapper.segment_frame(frame, prompt)

        if mask is None:
            raise RuntimeError(
                f"SAM2 frame inference returned None: "
                f"{self.wrapper.sam2_error_reason}"
            )

        elapsed_ms = (time.time() - t0) * 1000.0

        meta = {
            "backend_name": "sam2",
            "segmentation_method": "sam2",
            "ai_segmentation_used": True,
            "sam2_model_loaded": self.wrapper.sam2_model_loaded,
            "sam2_inference_ran": self.wrapper.sam2_inference_ran,
            "prompt_mode": prompt["prompt_mode"],
            "prompt_source": prompt["prompt_source"],
            "inference_ms": elapsed_ms,
            "fallback_used": False,
            "fallback_reason": None,
            "mask_confidence": prompt["confidence"],
        }

        return mask, meta
