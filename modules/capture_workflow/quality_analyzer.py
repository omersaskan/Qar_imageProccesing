import cv2
import numpy as np
from typing import Dict, Any, Optional
from .config import QualityThresholds, default_quality_thresholds


class QualityAnalyzer:
    def __init__(self, thresholds: QualityThresholds = default_quality_thresholds):
        self.thresholds = thresholds

    def get_blur_score(self, frame: np.ndarray) -> float:
        """
        Blur estimate using Laplacian variance.
        Higher = sharper.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def get_exposure_score(self, frame: np.ndarray) -> float:
        """
        Mean grayscale brightness.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return float(np.mean(gray))

    def _fallback_centroid_from_mask(self, mask: np.ndarray) -> Optional[Dict[str, float]]:
        M = cv2.moments(mask)
        if M["m00"] <= 1e-8:
            return None
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        return {"x": cx, "y": cy}

    def analyze_frame(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
        mask_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        blur_score = self.get_blur_score(frame)
        exposure_score = self.get_exposure_score(frame)

        is_blur_ok = bool(blur_score >= self.thresholds.min_blur_score)
        is_exposure_ok = bool(
            self.thresholds.min_exposure_score <= exposure_score <= self.thresholds.max_exposure_score
        )

        occupancy = 0.0
        center_dist = 1.0
        is_framed = False
        is_clipped = False
        fragment_count = 0
        largest_contour_ratio = 0.0
        solidity = 0.0
        mask_confidence = 0.0
        bbox = None

        if mask is not None:
            h, w = mask.shape[:2]
            white_pixels = np.sum(mask > 0)
            occupancy = float(white_pixels / max(h * w, 1))

            if mask_meta is not None:
                bbox = mask_meta.get("bbox")
                is_clipped = bool(mask_meta.get("is_clipped", False))
                fragment_count = int(mask_meta.get("fragment_count", 0))
                largest_contour_ratio = float(mask_meta.get("largest_contour_ratio", 0.0))
                solidity = float(mask_meta.get("solidity", 0.0))
                mask_confidence = float(
                    mask_meta.get("mask_confidence", mask_meta.get("confidence", 0.0))
                )
                centroid = mask_meta.get("centroid")
            else:
                centroid = self._fallback_centroid_from_mask(mask)

                edge_pixels = (
                    np.sum(mask[0, :] > 0)
                    + np.sum(mask[-1, :] > 0)
                    + np.sum(mask[:, 0] > 0)
                    + np.sum(mask[:, -1] > 0)
                )
                is_clipped = bool(edge_pixels > 0)

            if centroid is not None:
                dx = (float(centroid["x"]) - w / 2) / max(w / 2, 1.0)
                dy = (float(centroid["y"]) - h / 2) / max(h / 2, 1.0)
                center_dist = float(np.sqrt(dx ** 2 + dy ** 2))
                is_framed = bool(center_dist < 0.45)

        failure_reasons = []

        if not is_blur_ok:
            failure_reasons.append(f"blurry ({blur_score:.1f})")

        if not is_exposure_ok:
            failure_reasons.append(f"bad_exposure ({exposure_score:.1f})")

        if mask is not None:
            if occupancy < 0.05:
                failure_reasons.append(f"object_too_small ({occupancy:.2%})")

            if occupancy > 0.88:
                failure_reasons.append(f"object_too_large ({occupancy:.2%})")

            if not is_framed:
                failure_reasons.append(f"subject_not_centered (dist={center_dist:.2f})")

            if is_clipped:
                failure_reasons.append("subject_clipped")

            if mask_confidence < 0.45:
                failure_reasons.append(f"low_mask_confidence ({mask_confidence:.2f})")

            if fragment_count > 3:
                failure_reasons.append(f"fragmented_mask ({fragment_count})")

            if largest_contour_ratio > 0 and largest_contour_ratio < 0.75:
                failure_reasons.append(f"unstable_dominant_contour ({largest_contour_ratio:.2f})")

            if solidity > 0 and solidity < 0.50:
                failure_reasons.append(f"low_solidity ({solidity:.2f})")

        overall_pass = bool(
            is_blur_ok
            and is_exposure_ok
            and (mask is None or (0.05 < occupancy < 0.88))
            and (mask is None or is_framed)
            and not is_clipped
            and (mask is None or mask_confidence >= 0.45)
            and fragment_count <= 3
            and (largest_contour_ratio == 0.0 or largest_contour_ratio >= 0.75)
            and (solidity == 0.0 or solidity >= 0.50)
        )

        return {
            "blur_score": float(blur_score),
            "exposure_score": float(exposure_score),
            "occupancy": float(occupancy),
            "center_dist": float(center_dist),
            "bbox": bbox,
            "is_clipped": bool(is_clipped),
            "fragment_count": int(fragment_count),
            "largest_contour_ratio": float(largest_contour_ratio),
            "solidity": float(solidity),
            "mask_confidence": float(mask_confidence),
            "is_blur_ok": bool(is_blur_ok),
            "is_exposure_ok": bool(is_exposure_ok),
            "is_framed": bool(is_framed),
            "overall_pass": bool(overall_pass),
            "failure_reasons": failure_reasons,
        }

    def is_product_centered(self, frame: np.ndarray, mask: Optional[np.ndarray] = None, mask_meta: Optional[Dict[str, Any]] = None) -> bool:
        analysis = self.analyze_frame(frame, mask, mask_meta)
        return analysis["is_framed"]