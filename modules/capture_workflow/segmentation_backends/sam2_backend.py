"""
SAM2 Backend — Image-mode segmentation for ObjectMasker
========================================================

Uses SAM2Wrapper in image mode (SAM2ImagePredictor).
Does NOT use video predictor API.
"""

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
    Segmentation backend delegating to SAM2Wrapper (image mode).

    - If SAM2 is available + model loaded → real inference via
      SAM2Wrapper.segment_frame() (set_image / predict).
    - If unavailable → raises RuntimeError for ObjectMasker fallback.
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

        prompt = generate_prompts(
            frame_shape=(h, w),
            mode=settings.sam2_prompt_mode,
        )

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
            "sam2_mode": self.wrapper.sam2_mode,
            "temporal_consistency": self.wrapper.temporal_consistency,
            "api_type": self.wrapper.api_type,
            "prompt_mode": prompt["prompt_mode"],
            "prompt_source": prompt["prompt_source"],
            "inference_ms": elapsed_ms,
            "fallback_used": False,
            "fallback_reason": None,
            "mask_confidence": prompt["confidence"],
        }

        return mask, meta
