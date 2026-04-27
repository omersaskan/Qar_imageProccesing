"""
Depth Anything Wrapper — Scaffold Only
========================================

⚠️  NOT integrated into the reconstruction pipeline.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from modules.operations.settings import settings

logger = logging.getLogger(__name__)

HAS_DEPTH_ANYTHING = False
HAS_TORCH = False
DEPTH_IMPORT_ERROR_REASON: Optional[str] = None

if settings.depth_anything_enabled:
    try:
        import torch  # noqa: F401
        HAS_TORCH = True
    except ImportError:
        DEPTH_IMPORT_ERROR_REASON = "torch not installed"
    if HAS_TORCH:
        try:
            from depth_anything_v2.dpt import DepthAnythingV2  # noqa: F401
            HAS_DEPTH_ANYTHING = True
        except ImportError:
            DEPTH_IMPORT_ERROR_REASON = "depth-anything-v2 package not installed"
else:
    DEPTH_IMPORT_ERROR_REASON = "Depth Anything disabled (DEPTH_ANYTHING_ENABLED=false)"


class DepthAnythingWrapper:
    """Scaffold wrapper for Depth Anything v2. Not active in production."""

    def __init__(self):
        self.depth_enabled: bool = settings.depth_anything_enabled
        self.depth_available: bool = HAS_DEPTH_ANYTHING
        self.depth_model_loaded: bool = False
        self.depth_inference_ran: bool = False
        self.depth_error_reason: Optional[str] = DEPTH_IMPORT_ERROR_REASON
        self.device: str = settings.depth_anything_device if HAS_TORCH else "cpu"
        self.model_name: str = settings.depth_anything_model
        self.checkpoint: str = settings.depth_anything_checkpoint
        self.checkpoint_exists: bool = Path(self.checkpoint).exists() if self.checkpoint else False
        self.model = None

        if self.depth_available and not self.checkpoint_exists:
            self.depth_available = False
            self.depth_error_reason = f"Checkpoint not found at {self.checkpoint}"

    def is_available(self) -> bool:
        return self.depth_available

    def get_status(self) -> Dict[str, Any]:
        return {
            "depth_enabled": self.depth_enabled,
            "depth_available": self.depth_available,
            "depth_model_loaded": self.depth_model_loaded,
            "depth_inference_ran": self.depth_inference_ran,
            "depth_error_reason": self.depth_error_reason,
            "device": self.device,
            "checkpoint_exists": self.checkpoint_exists,
            "model_name": self.model_name,
            "checkpoint": self.checkpoint,
        }

    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Scaffold — not implemented."""
        if not self.is_available():
            return None
        logger.warning("Depth Anything inference not implemented (scaffold only)")
        return None
