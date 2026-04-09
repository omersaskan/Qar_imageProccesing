import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict

from .config import (
    QualityThresholds,
    ExtractionConfig,
    default_quality_thresholds,
    default_extraction_config,
)
from .quality_analyzer import QualityAnalyzer
from .object_masker import ObjectMasker
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("extractor")


class FrameExtractor:
    def __init__(
        self,
        quality_analyzer: Optional[QualityAnalyzer] = None,
        object_masker: Optional[ObjectMasker] = None,
        thresholds: Optional[QualityThresholds] = None,
        config: Optional[ExtractionConfig] = None,
    ):
        self.thresholds = (thresholds or default_quality_thresholds).model_copy(deep=True)
        self.config = (config or default_extraction_config).model_copy(deep=True)
        self.quality_analyzer = quality_analyzer or QualityAnalyzer(self.thresholds)
        self.object_masker = object_masker or ObjectMasker(thresholds=self.thresholds)

    def _apply_object_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        expanded_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        return cv2.bitwise_and(frame, frame, mask=expanded_mask)

    def _prepare_object_centric_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        bbox: Optional[Dict[str, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Suppress scene pixels outside an expanded object ROI without crop/resize.

        We intentionally keep the original image canvas and resolution intact.
        Per-frame crop normalization would change effective intrinsics between
        frames and can destabilize photogrammetry.
        """
        if bbox is None:
            return self._apply_object_mask(frame, mask), mask
        if self.config.roi_mode != "mask_suppression":
            raise ValueError(f"Unsupported ROI mode: {self.config.roi_mode}")

        h, w = mask.shape[:2]
        pad_x = max(8, int(bbox["w"] * self.config.roi_pad_x_ratio))
        pad_y = max(8, int(bbox["h"] * self.config.roi_pad_y_ratio))

        x1 = max(0, bbox["x"] - pad_x)
        y1 = max(0, bbox["y"] - pad_y)
        x2 = min(w, bbox["x"] + bbox["w"] + pad_x)
        y2 = min(h, bbox["y"] + bbox["h"] + pad_y)

        focus_mask = np.zeros_like(mask)
        focus_mask[y1:y2, x1:x2] = 255
        refined_mask = cv2.bitwise_and(mask, focus_mask)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        original_area = float(np.sum(mask > 0))
        refined_area = float(np.sum(refined_mask > 0))
        if original_area > 0 and refined_area < original_area * self.config.roi_min_retained_area_ratio:
            refined_mask = mask

        focused_frame = self._apply_object_mask(frame, refined_mask)
        return focused_frame, refined_mask

    def _write_verified_image(
        self,
        image_path: Path,
        image: np.ndarray,
        label: str,
        read_flag: int,
    ) -> None:
        write_ok = cv2.imwrite(str(image_path), image)
        if not write_ok:
            raise ValueError(f"{label} write failed: {image_path}")

        if not image_path.exists():
            raise ValueError(f"{label} missing after write: {image_path}")

        if image_path.stat().st_size <= 0:
            raise ValueError(f"{label} is empty after write: {image_path}")

        reloaded = cv2.imread(str(image_path), read_flag)
        if reloaded is None:
            try:
                image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
                if image_bytes.size > 0:
                    reloaded = cv2.imdecode(image_bytes, read_flag)
            except Exception:
                reloaded = None
        if reloaded is None or reloaded.size == 0:
            raise ValueError(f"{label} unreadable after write: {image_path}")

    def _bbox_iou(self, a: Dict[str, int], b: Dict[str, int]) -> float:
        ax1, ay1, aw, ah = a["x"], a["y"], a["w"], a["h"]
        bx1, by1, bw, bh = b["x"], b["y"], b["w"], b["h"]

        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter_area

        return float(inter_area / union) if union > 0 else 0.0

    def _get_masked_histogram(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask, [180, 128], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _should_reject_as_redundant(
        self,
        current_hist: np.ndarray,
        current_bbox: Optional[Dict[str, int]],
        last_hist: Optional[np.ndarray],
        last_bbox: Optional[Dict[str, int]],
    ) -> bool:
        if last_hist is None:
            return False

        hist_similarity = cv2.compareHist(last_hist, current_hist, cv2.HISTCMP_CORREL)

        bbox_iou = 0.0
        if current_bbox is not None and last_bbox is not None:
            bbox_iou = self._bbox_iou(last_bbox, current_bbox)

        return bool(
            hist_similarity > self.thresholds.min_similarity_score
            and bbox_iou > 0.85
        )

    def extract_keyframes(self, video_path: str, output_dir: str) -> List[str]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        extracted_paths: List[str] = []
        frame_count = 0

        last_extracted_hist = None
        last_bbox = None

        rejection_counts = {
            "sampling": 0,
            "quality_or_mask": 0,
            "redundant_similarity": 0,
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        masks_dir = output_path / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.thresholds.frame_sample_rate != 0:
                    rejection_counts["sampling"] += 1
                    frame_count += 1
                    continue

                mask, mask_meta = self.object_masker.generate_mask(frame)

                analysis = self.quality_analyzer.analyze_frame(frame, mask, mask_meta)
                if not analysis["overall_pass"]:
                    rejection_counts["quality_or_mask"] += 1
                    logger.debug(
                        f"Frame {frame_count} rejected :: reasons={analysis['failure_reasons']}"
                    )
                    frame_count += 1
                    continue

                current_hist = self._get_masked_histogram(frame, mask)
                current_bbox = mask_meta.get("bbox")

                if self._should_reject_as_redundant(
                    current_hist=current_hist,
                    current_bbox=current_bbox,
                    last_hist=last_extracted_hist,
                    last_bbox=last_bbox,
                ):
                    rejection_counts["redundant_similarity"] += 1
                    frame_count += 1
                    continue

                frame_filename = f"frame_{len(extracted_paths):04d}.jpg"
                frame_path = output_path / frame_filename
                mask_path = masks_dir / f"{frame_filename}.png"

                focused_frame, focused_mask = self._prepare_object_centric_frame(
                    frame,
                    mask,
                    current_bbox,
                )
                self._write_verified_image(frame_path, focused_frame, "Extracted frame", cv2.IMREAD_COLOR)
                self._write_verified_image(mask_path, focused_mask, "Mask", cv2.IMREAD_GRAYSCALE)

                extracted_paths.append(str(frame_path))
                last_extracted_hist = current_hist
                last_bbox = current_bbox

                if len(extracted_paths) >= self.config.max_frames:
                    break

                frame_count += 1
        finally:
            cap.release()

        logger.info(f"[Extraction] Product-aware summary for {Path(video_path).name}:")
        logger.info(f"   - Total frames read: {frame_count}")
        logger.info(f"   - Rejected by quality/mask: {rejection_counts['quality_or_mask']}")
        logger.info(f"   - Rejected by similarity: {rejection_counts['redundant_similarity']}")
        logger.info(f"   - Skipped by sampling: {rejection_counts['sampling']}")
        logger.info(f"   - Total saved: {len(extracted_paths)}")

        if not extracted_paths:
            logger.warning("Zero product-focused keyframes extracted.")

        return extracted_paths
