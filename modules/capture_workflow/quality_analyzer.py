import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .config import QualityThresholds, default_quality_thresholds

class QualityAnalyzer:
    def __init__(self, thresholds: QualityThresholds = default_quality_thresholds):
        self.thresholds = thresholds

    def analyze_frame(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze a single frame for blur, exposure and product quality.
        """
        blur_score = float(self.get_blur_score(frame))
        exposure_score = float(self.get_exposure_score(frame))
        
        is_blur_ok = bool(blur_score >= self.thresholds.min_blur_score)
        is_exposure_ok = bool(self.thresholds.min_exposure_score <= exposure_score <= self.thresholds.max_exposure_score)
        
        # Product-aware metrics
        occupancy = 0.0
        is_framed = False
        is_clipped = False
        center_dist = 1.0
        mask_confidence = 0.0
        largest_contour_ratio = 0.0
        fragment_count = 0
        
        if mask is not None:
            # Metadata from mask generation is helpful, but we can also re-calculate here
            white_pixels = np.sum(mask > 0)
            h, w = mask.shape[:2]
            occupancy = float(white_pixels / (h * w))
            
            # Re-verify mask characteristics for quality
            mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fragment_count = len(mask_contours)
            
            if mask_contours:
                main_c = max(mask_contours, key=cv2.contourArea)
                area = cv2.contourArea(main_c)
                largest_contour_ratio = float(area / white_pixels) if white_pixels > 0 else 0
                
                # Center of mass of the largest component (more robust than whole mask)
                M = cv2.moments(main_c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dx = (cx - w/2) / (w/2)
                    dy = (cy - h/2) / (h/2)
                    center_dist = float(np.sqrt(dx**2 + dy**2))
                    is_framed = bool(center_dist < 0.6)
            
            edge_pixels = np.sum(mask[0, :]) + np.sum(mask[-1, :]) + \
                          np.sum(mask[:, 0]) + np.sum(mask[:, -1])
            is_clipped = bool(edge_pixels > 0)
            
            # Confidence heuristic (simplified for analyzer)
            mask_confidence = 1.0
            if occupancy < 0.03 or occupancy > 0.95: mask_confidence *= 0.5
            if fragment_count > 5: mask_confidence *= 0.7
            if largest_contour_ratio < 0.7: mask_confidence *= 0.6
        else:
            # Fallback for if no mask is provided (less reliable)
            # Use exposure as a proxy for "something is there"
            is_framed = bool(10 < exposure_score < 245)

        failure_reasons = []
        if not is_blur_ok: failure_reasons.append(f"Blurry ({blur_score:.1f})")
        if not is_exposure_ok: failure_reasons.append(f"Exposure ({exposure_score:.1f})")
        if mask is not None:
            if occupancy < 0.03: failure_reasons.append(f"Object too small ({occupancy:.2%})")
            if occupancy > 0.95: failure_reasons.append(f"Object too large/filling frame ({occupancy:.2%})")
            if not is_framed: failure_reasons.append(f"Subject not centered (dist={center_dist:.2f})")
            if is_clipped: failure_reasons.append("Subject clipped at edges")
            if mask_confidence < 0.4: failure_reasons.append(f"Low mask confidence ({mask_confidence:.2f})")

        # Overall pass criteria: No failure reasons found
        return {
            "blur_score": blur_score,
            "exposure_score": exposure_score,
            "occupancy": occupancy,
            "center_dist": center_dist,
            "is_clipped": is_clipped,
            "mask_confidence": mask_confidence,
            "largest_contour_ratio": largest_contour_ratio,
            "fragment_count": fragment_count,
            "is_blur_ok": is_blur_ok,
            "is_exposure_ok": is_exposure_ok,
            "is_framed": is_framed,
            "overall_pass": len(failure_reasons) == 0,
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
        """Deprecated: Use analyze_frame with mask for better results."""
        analysis = self.analyze_frame(frame)
        return analysis["is_framed"]
