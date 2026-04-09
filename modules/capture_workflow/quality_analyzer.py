import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple

from .config import QualityThresholds, default_quality_thresholds


class QualityAnalyzer:
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = (thresholds or default_quality_thresholds).model_copy(deep=True)

    def get_blur_score(self, frame: np.ndarray) -> float:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def get_exposure_score(self, frame: np.ndarray) -> float:
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
        purity_score = 0.0
        border_touch_ratio = 0.0
        bottom_band_ratio = 0.0
        bottom_span_ratio = 0.0
        bottom_contact_ratio = 0.0
        support_area_ratio = 0.0
        support_suspected = False
        fallback_used = False
        backend_name = None
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
                purity_score = float(mask_meta.get("purity_score", 0.0))
                border_touch_ratio = float(mask_meta.get("border_touch_ratio", 0.0))
                bottom_band_ratio = float(mask_meta.get("bottom_band_ratio", 0.0))
                bottom_span_ratio = float(mask_meta.get("bottom_span_ratio", 0.0))
                bottom_contact_ratio = float(mask_meta.get("bottom_contact_ratio", 0.0))
                support_area_ratio = float(mask_meta.get("support_area_ratio", 0.0))
                support_suspected = bool(mask_meta.get("support_suspected", False))
                fallback_used = bool(mask_meta.get("fallback_used", False))
                backend_name = mask_meta.get("backend_name", "unknown")
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
                is_framed = bool(center_dist < self.thresholds.max_center_distance)

        failure_reasons = []

        if not is_blur_ok:
            failure_reasons.append(f"blurry ({blur_score:.1f})")

        if not is_exposure_ok:
            failure_reasons.append(f"bad_exposure ({exposure_score:.1f})")

        if mask is not None:
            if occupancy < self.thresholds.min_object_occupancy:
                failure_reasons.append(f"object_too_small ({occupancy:.2%})")

            if occupancy > self.thresholds.max_object_occupancy:
                failure_reasons.append(f"object_too_large ({occupancy:.2%})")

            if not is_framed:
                failure_reasons.append(f"subject_not_centered (dist={center_dist:.2f})")

            if is_clipped:
                failure_reasons.append("subject_clipped")

            if mask_confidence < self.thresholds.min_mask_confidence:
                failure_reasons.append(f"low_mask_confidence ({mask_confidence:.2f})")

            if purity_score < self.thresholds.min_mask_purity:
                failure_reasons.append(f"low_mask_purity ({purity_score:.2f})")

            if fragment_count > self.thresholds.max_mask_fragments:
                failure_reasons.append(f"fragmented_mask ({fragment_count})")

            if (
                largest_contour_ratio > 0
                and largest_contour_ratio < self.thresholds.min_dominant_contour_ratio
            ):
                failure_reasons.append(f"unstable_dominant_contour ({largest_contour_ratio:.2f})")

            if solidity > 0 and solidity < self.thresholds.min_mask_solidity:
                failure_reasons.append(f"low_solidity ({solidity:.2f})")

            if border_touch_ratio > self.thresholds.max_border_touch_ratio:
                failure_reasons.append(f"mask_touches_borders ({border_touch_ratio:.2f})")

            if fallback_used and mask_confidence < self.thresholds.min_mask_confidence:
                failure_reasons.append(f"fallback_mask_used_with_low_confidence ({mask_confidence:.2f})")

            # We don't blindly fail on fallback_used, but we record if it dropped below purity
            if fallback_used and purity_score < self.thresholds.min_mask_purity + 0.05:
                failure_reasons.append(f"fallback_mask_unstable_purity ({purity_score:.2f})")

            if support_suspected:
                failure_reasons.append("support_contamination_detected")

            if (
                bottom_band_ratio > self.thresholds.max_bottom_band_ratio
                and bottom_span_ratio > self.thresholds.max_bottom_span_ratio
            ):
                failure_reasons.append(
                    f"bottom_support_band ({bottom_band_ratio:.2f}/{bottom_span_ratio:.2f})"
                )

            if support_area_ratio > self.thresholds.max_support_area_ratio:
                failure_reasons.append(f"large_support_region ({support_area_ratio:.2f})")

        overall_pass = bool(
            is_blur_ok
            and is_exposure_ok
            and (
                mask is None
                or (
                    self.thresholds.min_object_occupancy
                    < occupancy
                    < self.thresholds.max_object_occupancy
                )
            )
            and (mask is None or is_framed)
            and not is_clipped
            and (mask is None or mask_confidence >= self.thresholds.min_mask_confidence)
            and (mask is None or purity_score >= self.thresholds.min_mask_purity)
            and fragment_count <= self.thresholds.max_mask_fragments
            and (
                largest_contour_ratio == 0.0
                or largest_contour_ratio >= self.thresholds.min_dominant_contour_ratio
            )
            and (solidity == 0.0 or solidity >= self.thresholds.min_mask_solidity)
            and (mask is None or border_touch_ratio <= self.thresholds.max_border_touch_ratio)
            and not support_suspected
            and (
                mask is None
                or not (
                    bottom_band_ratio > self.thresholds.max_bottom_band_ratio
                    and bottom_span_ratio > self.thresholds.max_bottom_span_ratio
                )
            )
            and (mask is None or support_area_ratio <= self.thresholds.max_support_area_ratio)
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
            "purity_score": float(purity_score),
            "border_touch_ratio": float(border_touch_ratio),
            "bottom_band_ratio": float(bottom_band_ratio),
            "bottom_span_ratio": float(bottom_span_ratio),
            "bottom_contact_ratio": float(bottom_contact_ratio),
            "support_area_ratio": float(support_area_ratio),
            "support_suspected": bool(support_suspected),
            "fallback_used": bool(fallback_used),
            "backend_name": backend_name,
            "is_blur_ok": bool(is_blur_ok),
            "is_exposure_ok": bool(is_exposure_ok),
            "is_framed": bool(is_framed),
            "overall_pass": bool(overall_pass),
            "failure_reasons": failure_reasons,
        }

    def is_product_centered(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
        mask_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        analysis = self.analyze_frame(frame, mask, mask_meta)
        return analysis["is_framed"]
