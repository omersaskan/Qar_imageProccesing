
import os
import logging
from typing import Optional, List, Dict, Any
import numpy as np
from .sam2_wrapper import SAM2Wrapper

logger = logging.getLogger(__name__)

class SegmentationMethod:
    LEGACY = "legacy"
    SAM2 = "sam2"

def get_segmentation_method() -> str:
    """Gets the segmentation method from environment variable or default."""
    return os.getenv("SEGMENTATION_METHOD", SegmentationMethod.LEGACY).lower()

def run_segmentation(video_path: str, method: Optional[str] = None) -> Optional[Dict[int, np.ndarray]]:
    """
    Runs segmentation using the specified method.
    If method is None, uses get_segmentation_method().
    Falls back to legacy if SAM2 fails or is unavailable.
    """
    if method is None:
        method = get_segmentation_method()
        
    if method == SegmentationMethod.SAM2:
        sam2 = SAM2Wrapper()
        if sam2.is_available():
            masks = sam2.segment_video(video_path, prompts=[])
            if masks is not None:
                return masks
            logger.warning("SAM2 segmentation returned None. Falling back to legacy.")
        else:
            logger.warning("SAM2 requested but not available. Falling back to legacy.")
            
    # Legacy logic would go here or be called from another module
    logger.info("Using legacy segmentation method.")
    return None
