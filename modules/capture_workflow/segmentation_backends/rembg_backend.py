import time
import cv2
import numpy as np
from typing import Tuple, Dict, Any
from modules.capture_workflow.config import SegmentationConfig

# We do not globally import rembg to avoid crashing the worker if not installed.
# We import inline in the class.

class RembgBackend:
    """
    ML-first segmentation backend using the rembg library (typically U-2-Net).
    It extracts the alpha channel as a semantic object mask.
    """

    def __init__(self):
        self._session = None
        self._model_name = None

    def _get_session(self, model_name: str):
        import rembg
        if self._session is None or self._model_name != model_name:
            self._session = rembg.new_session(model_name)
            self._model_name = model_name
        return self._session

    def segment(self, frame: np.ndarray, config: SegmentationConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
        import rembg
        start_time = time.perf_counter()
        
        session = self._get_session(config.rembg_model_name)
        
        # rembg expects RGB or BGRA etc. We pass the frame.
        # remove() usually handles numpy array, returns BGRA/RGBA. OpenCV is BGR.
        # rembg.remove handles color conversion internally but just in case:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        output = rembg.remove(rgb_frame, session=session, post_process_mask=True)
        
        # Defensive conversion: rembg can return PIL Image, bytes, or numpy array.
        # Typically returns RGBA numpy array when given Numpy.
        if hasattr(output, "convert"):
            output = np.array(output)
        elif isinstance(output, bytes):
            nparr = np.frombuffer(output, np.uint8)
            output = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
        if not hasattr(output, "shape"):
            raise ValueError(f"Unrecognized output format from rembg: {type(output)}")
            
        if len(output.shape) == 3 and output.shape[2] == 4:
            alpha = output[:, :, 3]
        elif len(output.shape) == 2:
            alpha = output
        else:
            raise ValueError(f"Unexpected output format from rembg: {output.shape}")

        binary = (alpha > config.rembg_mask_threshold).astype(np.uint8) * 255
        
        # Compute real confidence proxy based on the mean of non-zero alpha values
        positive_alpha = alpha[alpha > config.rembg_mask_threshold]
        if positive_alpha.size > 0:
            mask_confidence = float(np.mean(positive_alpha) / 255.0)
        else:
            mask_confidence = 0.0

        inference_ms = (time.perf_counter() - start_time) * 1000.0

        meta = {
            "backend_name": "rembg",
            "inference_ms": float(inference_ms),
            "fallback_used": False,
            "fallback_reason": None,
            "mask_confidence": mask_confidence,
            "purity_score": mask_confidence * 0.9, # base purity proxy
            "support_suspected": False
        }

        return binary, meta
