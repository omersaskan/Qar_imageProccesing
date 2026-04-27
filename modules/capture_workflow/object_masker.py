import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

from modules.operations.logging_config import get_component_logger
from .config import QualityThresholds, default_quality_thresholds, default_segmentation_config, SegmentationConfig
from .segmentation_backends.factory import BackendFactory

logger = get_component_logger("object_masker")


class ObjectMasker:
    """
    Standardizes object masking by calling a pluggable backend (e.g. rembg or heuristic),
    and enforcing common post-processing like support-band suppression and metric computation.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        thresholds: Optional[QualityThresholds] = None,
        config: Optional[SegmentationConfig] = None,
    ):
        self.use_gpu = use_gpu
        self.thresholds = (thresholds or default_quality_thresholds).model_copy(deep=True)
        self.config = (config or default_segmentation_config).model_copy(deep=True)

    # Post-processing utilities

    def _largest_component_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels <= 1:
            return binary_mask

        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return np.where(labels == largest_idx, 255, 0).astype(np.uint8)

    def _contour_metrics(self, binary_mask: np.ndarray) -> Dict[str, Any]:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "bbox": None,
                "centroid": None,
                "fragment_count": 0,
                "largest_contour_ratio": 0.0,
                "solidity": 0.0,
            }

        areas = [cv2.contourArea(c) for c in contours]
        total_area = float(sum(areas)) if areas else 0.0
        largest_idx = int(np.argmax(areas))
        largest = contours[largest_idx]
        largest_area = float(areas[largest_idx])

        x, y, w, h = cv2.boundingRect(largest)

        M = cv2.moments(largest)
        if M["m00"] > 1e-8:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx = float(x + w / 2)
            cy = float(y + h / 2)

        hull = cv2.convexHull(largest)
        hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
        solidity = (largest_area / hull_area) if hull_area > 1e-8 else 0.0

        return {
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "centroid": {"x": cx, "y": cy},
            "fragment_count": len(contours),
            "largest_contour_ratio": (largest_area / total_area) if total_area > 1e-8 else 0.0,
            "solidity": solidity,
        }

    # Removed legacy priority functions

    def _support_metrics(self, binary_mask: np.ndarray) -> Dict[str, float]:
        h, w = binary_mask.shape[:2]
        mask_area = float(np.sum(binary_mask > 0))
        if mask_area <= 0:
            return {
                "border_touch_ratio": 0.0,
                "bottom_band_ratio": 0.0,
                "bottom_span_ratio": 0.0,
                "bottom_contact_ratio": 0.0,
            }

        perimeter_sample = max(2.0 * w + 2.0 * h, 1.0)
        edge_pixels = (
            np.sum(binary_mask[0, :] > 0)
            + np.sum(binary_mask[-1, :] > 0)
            + np.sum(binary_mask[:, 0] > 0)
            + np.sum(binary_mask[:, -1] > 0)
        )

        bottom_band_height = max(8, int(h * 0.18))
        bottom_band = binary_mask[h - bottom_band_height:, :]
        bottom_band_ratio = float(np.sum(bottom_band > 0) / mask_area)

        bottom_contact_ratio = float(np.sum(binary_mask[-1, :] > 0) / max(w, 1))

        bottom_span_ratio = 0.0
        for row in bottom_band:
            cols = np.where(row > 0)[0]
            if cols.size == 0:
                continue
            row_span = float((cols.max() - cols.min() + 1) / max(w, 1))
            bottom_span_ratio = max(bottom_span_ratio, row_span)

        return {
            "border_touch_ratio": float(edge_pixels / perimeter_sample),
            "bottom_band_ratio": float(bottom_band_ratio),
            "bottom_span_ratio": float(bottom_span_ratio),
            "bottom_contact_ratio": float(bottom_contact_ratio),
        }

    def _remove_support_band(self, binary_mask: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        h, w = binary_mask.shape[:2]
        mask_area = float(np.sum(binary_mask > 0))
        if mask_area <= 0:
            return binary_mask, 0.0, False

        lower_only = np.zeros_like(binary_mask)
        lower_only[int(h * self.thresholds.support_scan_start_ratio):, :] = binary_mask[
            int(h * self.thresholds.support_scan_start_ratio):, :
        ]

        kernel_w = max(15, int(w * 0.18))
        kernel_h = max(3, int(h * 0.015))
        support_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        support = cv2.morphologyEx(lower_only, cv2.MORPH_OPEN, support_kernel)
        support = cv2.dilate(support, np.ones((3, 9), np.uint8), iterations=1)

        support_area = float(np.sum(support > 0))
        if support_area <= 0:
            return binary_mask, 0.0, False

        contours, _ = cv2.findContours(support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary_mask, 0.0, False

        largest = max(contours, key=cv2.contourArea)
        x, y, sw, sh = cv2.boundingRect(largest)
        span_ratio = float(sw / max(w, 1))
        height_ratio = float(sh / max(h, 1))
        area_ratio = float(support_area / mask_area)
        touches_bottom = (y + sh) >= (h - max(2, int(h * 0.02)))

        if not (
            span_ratio >= self.thresholds.support_min_span_ratio
            and height_ratio <= self.thresholds.support_max_height_ratio
            and area_ratio >= self.thresholds.support_min_area_ratio
            and (touches_bottom or y >= int(h * self.thresholds.support_min_y_ratio))
        ):
            return binary_mask, 0.0, False

        candidate = binary_mask.copy()
        candidate[support > 0] = 0
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        candidate = self._largest_component_mask(candidate)

        remaining_ratio = float(np.sum(candidate > 0) / mask_area)
        if remaining_ratio < self.thresholds.support_min_remaining_ratio:
            return binary_mask, 0.0, False

        return candidate, area_ratio, True

    def generate_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = frame.shape[:2]
        
        backend_name = self.config.backend
        if not self.config.enabled:
            backend_name = "heuristic"
            
        # Feature flag for SAM2 experiment
        env_method = os.getenv("SEGMENTATION_METHOD", "legacy").lower()
        if env_method == "sam2":
            # Check if SAM2 is available before switching
            from modules.ai_segmentation.sam2_wrapper import HAS_SAM2
            if HAS_SAM2:
                logger.info("SAM2 feature flag detected. Using sam2 backend.")
                backend_name = "sam2"
            else:
                logger.warning("SAM2 requested via feature flag but dependencies missing. Falling back to legacy.")

        backend = BackendFactory.get_backend(backend_name)
        
        try:
            binary, meta_out = backend.segment(frame, self.config)
        except Exception as e:
            logger.error(f"Backend '{backend_name}' failed: {e}")
            if self.config.hard_fail_on_backend_error:
                raise e
            if self.config.fallback_backend:
                logger.warning(f"Falling back to '{self.config.fallback_backend}'")
                backend = BackendFactory.get_backend(self.config.fallback_backend)
                binary, meta_out = backend.segment(frame, self.config)
                meta_out["fallback_used"] = True
                meta_out["fallback_reason"] = str(e)
            else:
                raise ValueError("No fallback backend defined and main backend failed.")

        # Post-processing to standardize format and suppress support band
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        binary = self._largest_component_mask(binary)

        binary, support_area_ratio, support_removed = self._remove_support_band(binary)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        binary = self._largest_component_mask(binary)

        metrics = self._contour_metrics(binary)
        support_metrics = self._support_metrics(binary)

        pixel_count = float(np.sum(binary > 0))
        occupancy = pixel_count / float(max(h * w, 1))

        is_clipped = bool(
            np.sum(binary[0, :] > 0)
            + np.sum(binary[-1, :] > 0)
            + np.sum(binary[:, 0] > 0)
            + np.sum(binary[:, -1] > 0)
            > 0
        )

        purity_score = (
            0.40 * float(metrics["largest_contour_ratio"])
            + 0.25 * float(metrics["solidity"])
            + 0.20 * float(max(0.0, 1.0 - min(1.0, support_area_ratio * 1.8)))
            + 0.15 * float(max(0.0, 1.0 - min(1.0, support_metrics["border_touch_ratio"] * 3.0)))
        )

        support_suspected = bool(
            support_metrics["bottom_band_ratio"] > self.thresholds.max_bottom_band_ratio
            or (
                support_metrics["bottom_span_ratio"] > self.thresholds.max_bottom_span_ratio
                and support_metrics["bottom_contact_ratio"] > self.thresholds.support_alert_bottom_contact_ratio
            )
            or (
                support_metrics["bottom_span_ratio"] > self.thresholds.support_alert_wide_span_ratio
                and support_metrics["bottom_band_ratio"] > self.thresholds.support_alert_wide_band_ratio
            )
            or support_area_ratio > self.thresholds.max_support_area_ratio
        )

        base_confidence = meta_out.get("mask_confidence", 0.10)
        confidence = base_confidence * 0.30  # weigh initial backend confidence
        confidence += min(0.30, float(metrics["largest_contour_ratio"]) * 0.30)
        confidence += min(0.20, float(metrics["solidity"]) * 0.20)
        confidence += min(
            0.15,
            float(
                max(
                    0.0,
                    self.thresholds.confidence_occupancy_tolerance
                    - abs(occupancy - self.thresholds.confidence_target_occupancy),
                )
            )
            / self.thresholds.confidence_occupancy_tolerance
            * 0.15,
        )
        confidence += min(0.15, purity_score * 0.15)
        confidence -= min(0.18, support_metrics["border_touch_ratio"] * 0.60)
        confidence -= min(0.18, support_metrics["bottom_band_ratio"] * 0.50)
        confidence -= min(0.18, support_area_ratio * 0.45)
        if is_clipped:
            confidence *= 0.70
        if metrics["fragment_count"] > 3:
            confidence *= 0.75
        confidence = float(np.clip(confidence, 0.0, 1.0))

        meta = {
            **meta_out,
            "confidence": confidence,
            "mask_confidence": confidence,
            "segmentation_method": backend_name,
            "purity_score": float(np.clip(purity_score, 0.0, 1.0)),
            "occupancy": float(occupancy),
            "is_clipped": is_clipped,
            "support_suspected": support_suspected,
            "support_removed": bool(support_removed),
            "support_area_ratio": float(support_area_ratio),
            **support_metrics,
            **metrics,
        }

        return binary, meta

    def process_session(self, frames_dir: str, masks_dir: str) -> List[Dict[str, Any]]:
        frames_path = Path(frames_dir)
        masks_path = Path(masks_dir)
        masks_path.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        frame_files = sorted(list(frames_path.glob("*.jpg")) + list(frames_path.glob("*.png")))

        for f_path in frame_files:
            frame = cv2.imread(str(f_path))
            if frame is None:
                continue

            mask, meta = self.generate_mask(frame)
            mask_out_path = masks_path / f"{f_path.name}.png"
            cv2.imwrite(str(mask_out_path), mask)

            meta["frame_name"] = f_path.name
            meta["mask_path"] = str(mask_out_path)
            results.append(meta)

            logger.debug(
                f"Mask generated for {f_path.name}: "
                f"occupancy={meta['occupancy']:.2f}, "
                f"confidence={meta['confidence']:.2f}, "
                f"purity={meta['purity_score']:.2f}, "
                f"support={meta['support_suspected']}"
            )

        return results
