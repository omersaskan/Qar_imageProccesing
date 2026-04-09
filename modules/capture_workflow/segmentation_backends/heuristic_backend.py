import time
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from modules.capture_workflow.config import SegmentationConfig

# We need the quality config thresholds for GrabCut seed thresholds.
# Rather than tightly coupling back to ObjectMasker constants, we fetch the module default config
from modules.capture_workflow.config import QualityThresholds, default_quality_thresholds

class HeuristicBackend:
    """
    Fallback backend using classical computer vision priors (Center, Edge, Contrast)
    with GrabCut refinement to isolate object foreground.
    """

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or default_quality_thresholds

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
        sx = max(w * 0.28, 1.0)
        sy = max(h * 0.28, 1.0)
        gauss = np.exp(-(((xs - cx) ** 2) / (2 * sx * sx) + ((ys - cy) ** 2) / (2 * sy * sy)))
        return self._normalize01(gauss)

    def _contrast_prior(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        border = np.concatenate(
            [lab[0, :, :], lab[-1, :, :], lab[:, 0, :], lab[:, -1, :]], axis=0
        )
        border_mean = border.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(lab - border_mean, axis=2)
        dist = cv2.GaussianBlur(dist, (0, 0), 3)
        return self._normalize01(dist)

    def _edge_prior(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.GaussianBlur(magnitude, (0, 0), 2)
        return self._normalize01(magnitude)

    def _largest_component_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels <= 1:
            return binary_mask
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return np.where(labels == largest_idx, 255, 0).astype(np.uint8)

    def _build_seed_mask(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        center_prior = self._center_prior(h, w)
        contrast_prior = self._contrast_prior(frame)
        edge_prior = self._edge_prior(frame)

        combined = 0.30 * center_prior + 0.45 * contrast_prior + 0.25 * edge_prior
        combined = self._normalize01(combined)

        seed = (combined > self.thresholds.seed_threshold).astype(np.uint8) * 255
        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        seed = self._largest_component_mask(seed)
        return seed

    def _grabcut_mask_init(self, frame: np.ndarray, seed: np.ndarray) -> np.ndarray:
        h, w = seed.shape[:2]
        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        border = int(max(4, min(h, w) * 0.05))
        gc_mask[:border, :] = cv2.GC_BGD
        gc_mask[-border:, :] = cv2.GC_BGD
        gc_mask[:, :border] = cv2.GC_BGD
        gc_mask[:, -border:] = cv2.GC_BGD

        sure_fg = cv2.erode(seed, np.ones((9, 9), np.uint8), iterations=1)
        prob_fg = cv2.dilate(seed, np.ones((11, 11), np.uint8), iterations=1)
        gc_mask[prob_fg > 0] = cv2.GC_PR_FGD
        gc_mask[sure_fg > 0] = cv2.GC_FGD

        contrast_prior = self._contrast_prior(frame)
        low_contrast_bg = (contrast_prior < self.thresholds.low_contrast_background_threshold).astype(np.uint8) * 255

        bottom_bg = np.zeros_like(seed)
        bottom_bg[int(h * self.thresholds.bottom_background_start_ratio):, :] = 255
        bottom_bg = cv2.bitwise_and(bottom_bg, low_contrast_bg)
        gc_mask[bottom_bg > 0] = cv2.GC_BGD

        return gc_mask

    def segment(self, frame: np.ndarray, config: SegmentationConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
        start_time = time.perf_counter()
        
        seed = self._build_seed_mask(frame)
        gc_mask = self._grabcut_mask_init(frame, seed)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(frame, gc_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
            binary = np.where(
                (gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD),
                0,
                255,
            ).astype(np.uint8)
        except Exception:
            binary = seed.copy()

        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        inference_ms = (time.perf_counter() - start_time) * 1000.0

        meta = {
            "backend_name": "heuristic",
            "inference_ms": float(inference_ms),
            "fallback_used": False,
            "fallback_reason": None,
            "mask_confidence": 0.5, # ObjectMasker post-processor corrects this
            "support_suspected": False # ObjectMasker post-processor overrides
        }

        return binary, meta
