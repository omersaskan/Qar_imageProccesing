import cv2
import numpy as np
from typing import Protocol, Tuple, Dict, Any, Optional
from modules.capture_workflow.config import SegmentationConfig

class SegmentationBackend(Protocol):
    def segment(self, frame: np.ndarray, config: SegmentationConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Segment a frame and return binary mask and metadata.
        Output metadata MUST include:
        - backend_name: str
        - inference_ms: float
        - fallback_used: bool
        - fallback_reason: Optional[str]
        - mask_confidence: float (normalized 0.0 - 1.0 proxy)
        """
        ...
