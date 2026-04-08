import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Optional
from .config import QualityThresholds, ExtractionConfig, default_quality_thresholds, default_extraction_config
from .quality_analyzer import QualityAnalyzer
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("extractor")

from .object_masker import ObjectMasker

logger = get_component_logger("extractor")

class FrameExtractor:
    def __init__(self, 
                 quality_analyzer: Optional[QualityAnalyzer] = None,
                 object_masker: Optional[ObjectMasker] = None,
                 thresholds: QualityThresholds = default_quality_thresholds,
                 config: ExtractionConfig = default_extraction_config):
        self.quality_analyzer = quality_analyzer or QualityAnalyzer(thresholds)
        self.object_masker = object_masker or ObjectMasker()
        self.thresholds = thresholds
        self.config = config

    def extract_keyframes(self, video_path: str, output_dir: str) -> List[str]:
        """
        Extracts high-quality, product-focused keyframes from a video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        extracted_paths = []
        frame_count = 0
        last_extracted_hist = None
        
        # Diagnostics
        rejection_counts = {
            "blur_or_exposure": 0,
            "object_isolation": 0,
            "similarity": 0,
            "sampling": 0
        }
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        masks_dir = output_path / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        crops_dir = output_path / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Time-based sampling
            if frame_count % self.thresholds.frame_sample_rate != 0:
                rejection_counts["sampling"] += 1
                frame_count += 1
                continue
            
            # 2. Object Masking
            mask, mask_meta = self.object_masker.generate_mask(frame)
            
            # 3. Quality filter (blur/exposure/product framing/mask confidence)
            analysis = self.quality_analyzer.analyze_frame(frame, mask)
            if not analysis["overall_pass"]:
                rejection_counts["object_isolation"] += 1
                reasons = ", ".join(analysis.get("failure_reasons", []))
                logger.debug(f"Frame {frame_count} rejected: {reasons}")
                frame_count += 1
                continue
            
            # 4. Similarity filter (avoid redundant frames) - Masked ROI
            current_hist = self._get_masked_histogram(frame, mask)
            if last_extracted_hist is not None:
                similarity = cv2.compareHist(last_extracted_hist, current_hist, cv2.HISTCMP_CORREL)
                if similarity > self.thresholds.min_similarity_score:
                    # Also check for bbox drift if available
                    rejection_counts["similarity"] += 1
                    frame_count += 1
                    continue
            
            # 5. Save frame and mask
            frame_filename = f"frame_{len(extracted_paths):04d}.jpg"
            frame_path = output_path / frame_filename
            mask_path = masks_dir / f"{frame_filename}.png"
            
            cv2.imwrite(str(frame_path), frame)
            cv2.imwrite(str(mask_path), mask)
            
            # Phase 2: Produce padded ROI crop
            bbox = mask_meta.get("bbox", (0, 0, w, h))
            crop = self._produce_padded_crop(frame, bbox)
            if crop is not None:
                crop_path = crops_dir / frame_filename
                cv2.imwrite(str(crop_path), crop)
            
            extracted_paths.append(str(frame_path))
            last_extracted_hist = current_hist
            
            # Stop if we reached max frames
            if len(extracted_paths) >= self.config.max_frames:
                break
            
            # Log progress every 20 frames read
            if frame_count % 20 == 0:
                logger.debug(f"Video {video_path} progress: Read {frame_count} frames, Extracted {len(extracted_paths)} product-focused keyframes.")
            
            frame_count += 1

        cap.release()
        
        # Diagnostic summary
        logger.info(f"📊 PRODUCT-AWARE EXTRACTION SUMMARY for {Path(video_path).name}:")
        logger.info(f"   - Total frames read: {frame_count}")
        logger.info(f"   - Rejected by quality/isolation: {rejection_counts['object_isolation']}")
        logger.info(f"   - Rejected by similarity: {rejection_counts['similarity']}")
        logger.info(f"   - Skipped by sampling: {rejection_counts['sampling']}")
        logger.info(f"   - ✅ TOTAL SAVED: {len(extracted_paths)}")

        if not extracted_paths:
            logger.warning(f"❌ ZERO product-focused keyframes extracted! Check quality settings or object visibility.")
        
        return extracted_paths

    def _get_masked_histogram(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Calculate a simple 3B histogram focused on the masked object."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Use only values where mask is white
        hist = cv2.calcHist([hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _produce_padded_crop(self, frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        """
        Produces a padded ROI crop.
        Padding: 12-15% of bbox dimensions.
        Clamped to image bounds, preserving aspect ratio.
        No upscaling, downscale only if long edge > 1600px.
        """
        h, w = frame.shape[:2]
        bx, by, bw, bh = bbox
        
        if bw == 0 or bh == 0:
            return None
            
        # 1. Padding (15%)
        pad_x = int(bw * 0.15)
        pad_y = int(bh * 0.15)
        
        x1, y1 = max(0, bx - pad_x), max(0, by - pad_y)
        x2, y2 = min(w, bx + bw + pad_x), min(h, by + bh + pad_y)
        
        crop = frame[y1:y2, x1:x2]
        
        # 2. Downscaling check (Long edge > 1536-1600px)
        ch, cw = crop.shape[:2]
        long_edge = max(ch, cw)
        target_limit = 1536
        
        if long_edge > target_limit:
            scale = target_limit / long_edge
            new_w, new_h = int(cw * scale), int(ch * scale)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        return crop
