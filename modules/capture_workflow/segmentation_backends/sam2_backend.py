
import numpy as np
import logging
from typing import Dict, Any, Tuple
from .base import SegmentationBackend
from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper

logger = logging.getLogger(__name__)


class SAM2Backend(SegmentationBackend):
    """
    Segmentation backend that delegates to the SAM2Wrapper.

    ⚠️  Real SAM2 frame segmentation is NOT yet implemented.
    This backend currently raises NotImplementedError, which is caught
    by ObjectMasker and triggers a safe fallback to the heuristic backend.

    The NotImplementedError is intentional and expected during the
    Phase 6.1 DEV-SUBSET hardening phase.
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

        # Prepare prompt based on settings
        h, w = frame.shape[:2]
        center_prompt = {"point": [w // 2, h // 2], "label": 1}

        # -----------------------------------------------------------------
        # STUB: Real SAM2 frame segmentation is NOT implemented.
        # This raise is intentional — ObjectMasker catches it and falls
        # back to the heuristic backend when hard_fail_on_backend_error
        # is False (the default).
        #
        # Implementation requires user review approval.
        # -----------------------------------------------------------------
        raise NotImplementedError(
            "SAM2 frame segmentation not yet implemented in backend. "
            "Fallback to legacy is expected."
        )
