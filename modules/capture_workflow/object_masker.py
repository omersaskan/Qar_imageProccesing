import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("object_masker")


class ObjectMasker:
    """
    Product-focused masking using multiple priors:
    - center prior
    - border contrast prior
    - GrabCut refinement
    - contour-driven quality metrics
    """

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu

    def _normalize01(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - mn) / (mx - mn)

    def _center_prior(self, h: int, w: int) -> np.ndarray:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = w / 2.0
        cy = h / 2.0

        sx = max(w * 0.22, 1.0)
        sy = max(h * 0.22, 1.0)

        gauss = np.exp(-(((xs - cx) ** 2) / (2 * sx * sx) + ((ys - cy) ** 2) / (2 * sy * sy)))
        return self._normalize01(gauss)

    def _contrast_prior(self, frame: np.ndarray) -> np.ndarray:
        """
        Lightweight saliency approximation by comparing pixels
        to the average border color in Lab space.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        border = np.concatenate(
            [
                lab[0, :, :],
                lab[-1, :, :],
                lab[:, 0, :],
                lab[:, -1, :],
            ],
            axis=0,
        )

        border_mean = border.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(lab - border_mean, axis=2)
        dist = cv2.GaussianBlur(dist, (0, 0), 3)
        return self._normalize01(dist)

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

    def _build_seed_mask(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        center_prior = self._center_prior(h, w)
        contrast_prior = self._contrast_prior(frame)

        combined = 0.60 * center_prior + 0.40 * contrast_prior
        combined = self._normalize01(combined)

        seed = (combined > 0.45).astype(np.uint8) * 255
        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        seed = self._largest_component_mask(seed)
        return seed

    def _seed_to_rect(self, seed: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = seed.shape[:2]
        ys, xs = np.where(seed > 0)

        if len(xs) > 50 and len(ys) > 50:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            pad_x = int(0.12 * max(1, (x2 - x1)))
            pad_y = int(0.12 * max(1, (y2 - y1)))

            x = max(0, x1 - pad_x)
            y = max(0, y1 - pad_y)
            rw = min(w - x, (x2 - x1) + 2 * pad_x)
            rh = min(h - y, (y2 - y1) + 2 * pad_y)
            return (int(x), int(y), int(rw), int(rh))

        margin_h = int(h * 0.15)
        margin_w = int(w * 0.15)
        return (margin_w, margin_h, w - 2 * margin_w, h - 2 * margin_h)

    def generate_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = frame.shape[:2]

        seed = self._build_seed_mask(frame)
        rect = self._seed_to_rect(seed)

        gc_mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(frame, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            binary = np.where(
                (gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD),
                0,
                255,
            ).astype(np.uint8)
        except Exception as e:
            logger.error(f"GrabCut failed: {e}")
            binary = seed.copy()

        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        binary = self._largest_component_mask(binary)

        metrics = self._contour_metrics(binary)

        pixel_count = float(np.sum(binary > 0))
        occupancy = pixel_count / float(max(h * w, 1))

        edge_pixels = (
            np.sum(binary[0, :] > 0)
            + np.sum(binary[-1, :] > 0)
            + np.sum(binary[:, 0] > 0)
            + np.sum(binary[:, -1] > 0)
        )
        is_clipped = bool(edge_pixels > 0)

        confidence = 1.0

        if occupancy < 0.04 or occupancy > 0.85:
            confidence *= 0.35
        if is_clipped:
            confidence *= 0.70
        if metrics["fragment_count"] > 3:
            confidence *= 0.75
        if metrics["largest_contour_ratio"] < 0.75:
            confidence *= 0.70
        if metrics["solidity"] < 0.50:
            confidence *= 0.80

        meta = {
            "confidence": float(confidence),
            "mask_confidence": float(confidence),
            "occupancy": float(occupancy),
            "is_clipped": is_clipped,
            "rect": rect,
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
                f"fragments={meta['fragment_count']}, "
                f"largest_ratio={meta['largest_contour_ratio']:.2f}"
            )

        return results