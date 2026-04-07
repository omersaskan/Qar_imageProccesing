import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Optional
from .config import QualityThresholds, ExtractionConfig, default_quality_thresholds, default_extraction_config
from .quality_analyzer import QualityAnalyzer
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("extractor")

class FrameExtractor:
    def __init__(self, 
                 quality_analyzer: Optional[QualityAnalyzer] = None,
                 thresholds: QualityThresholds = default_quality_thresholds,
                 config: ExtractionConfig = default_extraction_config):
        self.quality_analyzer = quality_analyzer or QualityAnalyzer(thresholds)
        self.thresholds = thresholds
        self.config = config

    def extract_keyframes(self, video_path: str, output_dir: str) -> List[str]:
        """
        Extracts high-quality, diverse keyframes from a video.
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
            "similarity": 0,
            "sampling": 0
        }
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Time-based sampling
            if frame_count % self.thresholds.frame_sample_rate != 0:
                rejection_counts["sampling"] += 1
                frame_count += 1
                continue
            
            # 2. Quality filter (blur/exposure)
            analysis = self.quality_analyzer.analyze_frame(frame)
            if not analysis["overall_pass"]:
                rejection_counts["blur_or_exposure"] += 1
                reasons = ", ".join(analysis.get("failure_reasons", []))
                logger.debug(f"Frame {frame_count} rejected: {reasons}")
                frame_count += 1
                continue
            
            # 3. Similarity filter (avoid redundant frames)
            current_hist = self._get_histogram(frame)
            if last_extracted_hist is not None:
                similarity = cv2.compareHist(last_extracted_hist, current_hist, cv2.HISTCMP_CORREL)
                if similarity > self.thresholds.min_similarity_score:
                    rejection_counts["similarity"] += 1
                    frame_count += 1
                    continue
            
            # 4. Save frame
            frame_filename = f"frame_{len(extracted_paths):04d}.jpg"
            frame_path = output_path / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            extracted_paths.append(str(frame_path))
            last_extracted_hist = current_hist
            
            # Stop if we reached max frames
            if len(extracted_paths) >= self.config.max_frames:
                break
            
            # Log progress every 20 frames read
            if frame_count % 20 == 0:
                logger.debug(f"Video {video_path} progress: Read {frame_count} frames, Extracted {len(extracted_paths)} valid keyframes.")
            
            frame_count += 1

        cap.release()
        
        # Diagnostic summary
        logger.info(f"📊 EXTRACTION SUMMARY for {Path(video_path).name}:")
        logger.info(f"   - Total frames read: {frame_count}")
        logger.info(f"   - Rejected by quality (blur/exposure): {rejection_counts['blur_or_exposure']}")
        logger.info(f"   - Rejected by similarity: {rejection_counts['similarity']}")
        logger.info(f"   - Skipped by sampling: {rejection_counts['sampling']}")
        logger.info(f"   - ✅ TOTAL SAVED: {len(extracted_paths)}")

        if not extracted_paths:
            logger.warning(f"❌ ZERO valid keyframes extracted from {video_path}! Check quality settings or video content.")
        
        return extracted_paths

    def _get_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Calculate a simple 3B histogram for similarity comparison."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
