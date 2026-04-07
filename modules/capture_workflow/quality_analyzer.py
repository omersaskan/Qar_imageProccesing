import cv2
import numpy as np
from typing import Dict, Any, Tuple
from .config import QualityThresholds, default_quality_thresholds

class QualityAnalyzer:
    def __init__(self, thresholds: QualityThresholds = default_quality_thresholds):
        self.thresholds = thresholds

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a single frame for blur, exposure and framing.
        """
        blur_score = float(self.get_blur_score(frame))
        exposure_score = float(self.get_exposure_score(frame))
        
        is_blur_ok = bool(blur_score >= self.thresholds.min_blur_score)
        is_exposure_ok = bool(self.thresholds.min_exposure_score <= exposure_score <= self.thresholds.max_exposure_score)
        
        is_framed = bool(self.is_product_centered(frame))

        failure_reasons = []
        if not is_blur_ok: failure_reasons.append(f"Blurry ({blur_score:.1f} < {self.thresholds.min_blur_score})")
        if not is_exposure_ok: failure_reasons.append(f"Exposure ({exposure_score:.1f} outside range)")
        if not is_framed: failure_reasons.append("Subject not centered")

        return {
            "blur_score": blur_score,
            "exposure_score": exposure_score,
            "is_blur_ok": is_blur_ok,
            "is_exposure_ok": is_exposure_ok,
            "is_framed": is_framed,
            "overall_pass": bool(is_blur_ok and is_exposure_ok and is_framed),
            "failure_reasons": failure_reasons
        }

    def get_blur_score(self, frame: np.ndarray) -> float:
        """Calculate the Laplacian variance as a blur metric."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def get_exposure_score(self, frame: np.ndarray) -> float:
        """Calculate the average brightness as an exposure metric."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return float(np.mean(gray))

    def is_product_centered(self, frame: np.ndarray) -> bool:
        """Heuristic to check if the product is centered."""
        # Simple implementation: divide frame into 3x3 grid and check central grid
        # For MVP, we assume it's true if the frame is not completely dark/white
        brightness = self.get_exposure_score(frame)
        return 10 < brightness < 245
