import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("object_masker")

class ObjectMasker:
    """
    OpenCV-based object masking for product isolation.
    Uses Saliency + GrabCut with a center-weighted prior.
    """
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu

    def generate_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generates a binary mask for the main object in the frame.
        Returns (mask, metadata).
        """
        h, w = frame.shape[:2]
        
        # 1. Background Subtraction / Saliency Prior
        # For simplicity and robustness, we assume the object is roughly centered.
        # We create a rect that covers the central 70% of the image.
        margin_h = int(h * 0.15)
        margin_w = int(w * 0.15)
        rect = (margin_w, margin_h, w - 2 * margin_w, h - 2 * margin_h)
        
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # 2. GrabCut execution
        try:
            cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        except Exception as e:
            logger.error(f"GrabCut failed: {e}")
            return np.zeros((h, w), np.uint8), {"confidence": 0.0, "occupancy": 0.0}

        # 3. Post-process mask
        # 0 & 2 are background, 1 & 3 are foreground
        binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # 4. Refine mask (Morphology)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # 5. Calculate Metrics
        pixel_count = np.sum(binary_mask)
        occupancy = float(pixel_count / (h * w))
        
        # Confidence heuristic: based on occupancy and centrality
        # If occupancy is too low or too high (covering entire frame), confidence drops.
        confidence = 1.0
        if occupancy < 0.05 or occupancy > 0.9:
            confidence = 0.3
        
        # Check if mask is clipped at edges
        edge_pixels = np.sum(binary_mask[0, :]) + np.sum(binary_mask[-1, :]) + \
                      np.sum(binary_mask[:, 0]) + np.sum(binary_mask[:, -1])
        if edge_pixels > 0:
            confidence *= 0.7 # Clipped object reduces confidence
            
        return binary_mask * 255, {
            "confidence": confidence,
            "occupancy": occupancy,
            "is_clipped": edge_pixels > 0,
            "rect": rect
        }

    def process_session(self, frames_dir: str, masks_dir: str) -> List[Dict[str, Any]]:
        """
        Batch process all frames in a session.
        """
        frames_path = Path(frames_dir)
        masks_path = Path(masks_dir)
        masks_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        frame_files = sorted(list(frames_path.glob("*.jpg")) + list(frames_path.glob("*.png")))
        
        for f_path in frame_files:
            frame = cv2.imread(str(f_path))
            if frame is None:
                continue
                
            mask, meta = self.generate_mask(frame)
            
            mask_filename = f_path.name # COLMAP expected format or matched names
            mask_out_path = masks_path / mask_filename
            cv2.imwrite(str(mask_out_path), mask)
            
            meta["frame_name"] = f_path.name
            meta["mask_path"] = str(mask_out_path)
            results.append(meta)
            
            logger.debug(f"Mask generated for {f_path.name}: occupancy={meta['occupancy']:.2f}, confidence={meta['confidence']:.2f}")
            
        return results
