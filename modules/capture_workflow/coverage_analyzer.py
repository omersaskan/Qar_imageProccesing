from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import json
import numpy as np

from .config import CoverageConfig, default_coverage_config


class CoverageAnalyzer:
    """
    Estimates whether the extracted frame set has enough viewpoint diversity to
    justify reconstruction.

    This remains heuristic, but the thresholds are now explicit configuration
    rather than being hardcoded into the worker path.
    """

    def __init__(self, config: Optional[CoverageConfig] = None):
        self.config = (config or default_coverage_config).model_copy(deep=True)

    def _read_image(self, image_path: Path, read_flag: int):
        image = cv2.imread(str(image_path), read_flag)
        if image is not None:
            return image

        try:
            image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
            if image_bytes.size == 0:
                return None
            return cv2.imdecode(image_bytes, read_flag)
        except Exception:
            return None

    def _mask_path_for_frame(self, frame_path: Path) -> Path:
        return frame_path.parent / "masks" / f"{frame_path.name}.png"

    def _meta_path_for_frame(self, frame_path: Path) -> Path:
        return frame_path.parent / "masks" / f"{frame_path.name}.json"

    def _extract_signature(self, frame_path: Path) -> Optional[Dict[str, Any]]:
        frame = self._read_image(frame_path, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return None

        mask_path = self._mask_path_for_frame(frame_path)
        mask = self._read_image(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or np.sum(mask > 0) <= 0:
            return None

        ys, xs = np.where(mask > 0)
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        bbox_w = max(1, x2 - x1 + 1)
        bbox_h = max(1, y2 - y1 + 1)

        h, w = mask.shape[:2]
        area_ratio = float(np.sum(mask > 0) / max(h * w, 1))
        center_x = float((x1 + x2) / 2.0 / max(w, 1))
        center_y = float((y1 + y2) / 2.0 / max(h, 1))
        aspect_ratio = float(bbox_w / max(bbox_h, 1))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask, [64, 64], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        moments = cv2.moments(mask)
        hu = cv2.HuMoments(moments).flatten()
        hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
        
        fallback_used = False
        confidence = 1.0
        meta_path = self._meta_path_for_frame(frame_path)
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    fallback_used = meta.get("fallback_used", False)
                    confidence = meta.get("mask_confidence", confidence)
            except Exception:
                pass

        return {
            "path": str(frame_path),
            "hist": hist,
            "center": (center_x, center_y),
            "area_ratio": area_ratio,
            "aspect_ratio": aspect_ratio,
            "hu": hu,
            "fallback_used": fallback_used,
            "confidence": confidence,
        }

    def _is_new_view(self, signature: Dict[str, Any], representatives: List[Dict[str, Any]]) -> bool:
        for rep in representatives:
            hist_similarity = cv2.compareHist(rep["hist"], signature["hist"], cv2.HISTCMP_CORREL)
            center_dist = float(
                np.linalg.norm(
                    np.array(rep["center"], dtype=np.float32)
                    - np.array(signature["center"], dtype=np.float32)
                )
            )
            area_delta = abs(float(rep["area_ratio"]) - float(signature["area_ratio"]))
            aspect_delta = abs(float(rep["aspect_ratio"]) - float(signature["aspect_ratio"]))
            hu_delta = float(np.mean(np.abs(rep["hu"] - signature["hu"])))

            if (
                hist_similarity > self.config.dedupe_hist_similarity
                and center_dist < self.config.dedupe_center_distance_max
                and area_delta < self.config.dedupe_area_delta_max
                and aspect_delta < self.config.dedupe_aspect_delta_max
                and hu_delta < self.config.dedupe_hu_delta_max
            ):
                return False

        return True

    def analyze_coverage(self, extracted_frames: List[str]) -> Dict[str, Any]:
        signatures: List[Dict[str, Any]] = []
        unreadable_frames = 0
        fallback_frames = 0
        low_confidence_frames = 0

        for frame_str in extracted_frames:
            sig = self._extract_signature(Path(frame_str))
            if sig is None:
                unreadable_frames += 1
                continue
            if sig.get("fallback_used", False):
                fallback_frames += 1
            if sig.get("confidence", 1.0) < 0.6:
                low_confidence_frames += 1
            signatures.append(sig)

        num_frames = len(extracted_frames)
        readable_frames = len(signatures)
        reasons = []

        if readable_frames == 0:
            return {
                "num_frames": num_frames,
                "readable_frames": 0,
                "unique_views": 0,
                "diversity": "insufficient",
                "top_down_captured": False,
                "coverage_score": 0.0,
                "overall_status": "insufficient",
                "recommended_action": "needs_recapture",
                "reasons": ["No readable masked frames available for coverage analysis."],
            }

        representatives: List[Dict[str, Any]] = []
        centers_x: List[float] = []
        centers_y: List[float] = []
        areas: List[float] = []
        aspects: List[float] = []

        for sig in signatures:
            centers_x.append(sig["center"][0])
            centers_y.append(sig["center"][1])
            areas.append(sig["area_ratio"])
            aspects.append(sig["aspect_ratio"])
            if self._is_new_view(sig, representatives):
                representatives.append(sig)

        unique_views = len(representatives)
        center_x_span = float(max(centers_x) - min(centers_x)) if centers_x else 0.0
        center_y_span = float(max(centers_y) - min(centers_y)) if centers_y else 0.0
        scale_variation = float(max(areas) / max(min(areas), 1e-6)) if areas else 1.0
        aspect_variation = float(max(aspects) - min(aspects)) if aspects else 0.0

        if readable_frames < self.config.min_readable_frames:
            reasons.append(
                f"Too few readable object-centric frames ({readable_frames}/{self.config.min_readable_frames})."
            )

        if unique_views < self.config.min_unique_views:
            reasons.append(
                f"Insufficient viewpoint diversity ({unique_views}/{self.config.min_unique_views} unique views)."
            )

        if (
            center_x_span < self.config.min_center_x_span
            and center_y_span < self.config.min_center_y_span
        ):
            reasons.append("Object motion across accepted views is too narrow.")

        if (
            scale_variation < self.config.min_scale_variation
            and aspect_variation < self.config.min_aspect_variation
        ):
            reasons.append("Accepted views show too little scale/shape variation.")

        if unreadable_frames > 0:
            reasons.append(f"{unreadable_frames} extracted frames could not be analyzed.")
            
        if fallback_frames > max(1, readable_frames * 0.5):
            reasons.append(f"Too many frames relied on heuristic fallback ({fallback_frames}).")
            
        if low_confidence_frames > max(1, readable_frames * 0.3):
            reasons.append(f"Too many frames have low semantic confidence ({low_confidence_frames}).")

        coverage_score = (
            min(1.0, readable_frames / max(self.config.min_readable_frames, 1)) * 0.30
            + min(1.0, unique_views / max(self.config.min_unique_views, 1)) * 0.40
            + min(
                1.0,
                max(
                    center_x_span / self.config.span_score_x_target,
                    center_y_span / self.config.span_score_y_target,
                ),
            )
            * 0.15
            + min(
                1.0,
                max(
                    (scale_variation - 1.0) / self.config.scale_score_target,
                    aspect_variation / self.config.aspect_score_target,
                ),
            )
            * 0.15
        )

        diversity_status = "sufficient" if unique_views >= self.config.min_unique_views else "insufficient"
        elevated_view_proxy = bool(
            scale_variation >= self.config.elevated_view_scale_variation
            or aspect_variation >= self.config.elevated_view_aspect_variation
            or center_y_span >= self.config.elevated_view_center_y_span
        )
        overall_status = "sufficient" if not reasons else "insufficient"

        return {
            "num_frames": num_frames,
            "readable_frames": readable_frames,
            "unique_views": unique_views,
            "diversity": diversity_status,
            "top_down_captured": elevated_view_proxy,
            "center_x_span": center_x_span,
            "center_y_span": center_y_span,
            "scale_variation": scale_variation,
            "aspect_variation": aspect_variation,
            "coverage_score": float(np.clip(coverage_score, 0.0, 1.0)),
            "fallback_frames": fallback_frames,
            "low_confidence_frames": low_confidence_frames,
            "overall_status": overall_status,
            "recommended_action": "reconstruct" if overall_status == "sufficient" else "needs_recapture",
            "reasons": reasons,
        }
