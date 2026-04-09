import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from .config import QualityThresholds, ExtractionConfig, default_quality_thresholds, default_extraction_config
from .quality_analyzer import QualityAnalyzer
from .object_masker import ObjectMasker
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("extractor")


class FrameExtractor:
    def __init__(
        self,
        quality_analyzer: Optional[QualityAnalyzer] = None,
        object_masker: Optional[ObjectMasker] = None,
        thresholds: QualityThresholds = default_quality_thresholds,
        config: ExtractionConfig = default_extraction_config,
    ):
        self.quality_analyzer = quality_analyzer or QualityAnalyzer(thresholds)
        self.object_masker = object_masker or ObjectMasker()
        self.thresholds = thresholds
        self.config = config

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

    def extract_keyframes(self, video_path: str, output_dir: str) -> List[str]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        extracted_paths: List[str] = []
        frame_count = 0
        last_extracted_hist = None
        last_bbox = None

        rejection_counts = {
            "quality_or_mask": 0,
            "similarity": 0,
            "sampling": 0,
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        masks_dir = output_path / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

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
                logger.debug(f"Frame {frame_count} rejected: {analysis['failure_reasons']}")
                frame_count += 1
                continue

            current_hist = self._get_masked_histogram(frame, mask)
            current_bbox = mask_meta.get("bbox")

            if last_extracted_hist is not None:
                hist_similarity = cv2.compareHist(last_extracted_hist, current_hist, cv2.HISTCMP_CORREL)
                bbox_iou = self._bbox_iou(last_bbox, current_bbox) if (last_bbox and current_bbox) else 0.0

                # only reject if both appearance and ROI geometry are too similar
                if hist_similarity > self.thresholds.min_similarity_score and bbox_iou > 0.85:
                    rejection_counts["similarity"] += 1
                    frame_count += 1
                    continue

            frame_filename = f"frame_{len(extracted_paths):04d}.jpg"
            frame_path = output_path / frame_filename
            mask_path = masks_dir / f"{frame_filename}.png"

            cv2.imwrite(str(frame_path), frame)
            cv2.imwrite(str(mask_path), mask)

            extracted_paths.append(str(frame_path))
            last_extracted_hist = current_hist
            last_bbox = current_bbox

            if len(extracted_paths) >= self.config.max_frames:
                break

            frame_count += 1

        cap.release()

        logger.info(f"📊 PRODUCT-AWARE EXTRACTION SUMMARY for {Path(video_path).name}:")
        logger.info(f"   - Total frames read: {frame_count}")
        logger.info(f"   - Rejected by quality/mask: {rejection_counts['quality_or_mask']}")
        logger.info(f"   - Rejected by similarity: {rejection_counts['similarity']}")
        logger.info(f"   - Skipped by sampling: {rejection_counts['sampling']}")
        logger.info(f"   - ✅ TOTAL SAVED: {len(extracted_paths)}")

        if not extracted_paths:
            logger.warning("❌ ZERO product-focused keyframes extracted!")

        return extracted_paths