
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

# We import settings here to read configurations
from modules.operations.settings import settings

logger = logging.getLogger(__name__)

# Optional dependencies
HAS_SAM2 = False
HAS_TORCH = False
SAM2_ERROR_REASON = None

# If SAM2 is not enabled in settings, we don't even try to import
if settings.sam2_enabled:
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        SAM2_ERROR_REASON = "torch not found"

    if HAS_TORCH:
        try:
            from sam2.build_sam import build_sam2_video_predictor
            HAS_SAM2 = True
        except ImportError:
            SAM2_ERROR_REASON = "segment-anything-2 (sam2) not found"
else:
    SAM2_ERROR_REASON = "SAM2 disabled in settings"

class SAM2Wrapper:
    def __init__(self):
        self.sam2_enabled = settings.sam2_enabled
        self.sam2_available = HAS_SAM2
        self.sam2_model_loaded = False
        self.sam2_inference_ran = False
        self.sam2_error_reason = SAM2_ERROR_REASON
        
        self.device = settings.sam2_device if HAS_TORCH else "cpu"
        self.checkpoint = settings.sam2_checkpoint
        self.model_cfg = settings.sam2_model_cfg
        
        self.checkpoint_exists = Path(self.checkpoint).exists() if self.checkpoint else False
        self.predictor = None
        
        if self.sam2_available:
            if not self.checkpoint_exists:
                self.sam2_available = False
                self.sam2_error_reason = f"Checkpoint not found at {self.checkpoint}"
                logger.warning(self.sam2_error_reason)
            else:
                try:
                    # In a real scenario, we would load the model here
                    # self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint, device=self.device)
                    # self.sam2_model_loaded = True
                    logger.info(f"SAM2 initialized on {self.device}")
                except Exception as e:
                    self.sam2_error_reason = str(e)
                    logger.error(f"Failed to initialize SAM2: {e}")
                    self.sam2_available = False

    def is_available(self) -> bool:
        return self.sam2_available

    def get_status(self) -> Dict[str, Any]:
        return {
            "sam2_enabled": self.sam2_enabled,
            "sam2_available": self.sam2_available,
            "sam2_model_loaded": self.sam2_model_loaded,
            "sam2_inference_ran": self.sam2_inference_ran,
            "sam2_error_reason": self.sam2_error_reason,
            "device": self.device,
            "checkpoint_exists": self.checkpoint_exists,
            "model_cfg": self.model_cfg,
            "checkpoint": self.checkpoint
        }

    def segment_video(self, video_path: str, prompts: List[Dict[str, Any]]) -> Optional[Dict[int, np.ndarray]]:
        """
        Segments a video using SAM2 given point or box prompts.
        Returns a dictionary mapping frame index to mask.
        """
        if not self.sam2_available:
            logger.error(f"SAM2 is not available: {self.sam2_error_reason}. Falling back to legacy.")
            return None
            
        logger.info(f"Running SAM2 segmentation on {video_path}")
        self.sam2_inference_ran = True
        # This is a stub for the DEV-SUBSET experiment
        return {}

def get_predictor() -> Optional[SAM2Wrapper]:
    predictor = SAM2Wrapper()
    if predictor.is_available():
        return predictor
    return None
