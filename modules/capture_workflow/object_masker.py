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
        Generates a binary mask for the main object in the frame using multi-cue priors.
        Returns (mask, metadata).
        """
        h, w = frame.shape[:2]
        
        # Phase 2: Multi-cue priors
        # 1. Saliency Prior (with fallback)
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            (success, saliency_map) = saliency.computeSaliency(frame)
            saliency_map = (saliency_map * 255).astype("uint8")
        except AttributeError:
            # Fallback: Simple contrast-based saliency
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            saliency_map = cv2.absdiff(gray, cv2.GaussianBlur(gray, (21, 21), 0))
            saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
        
        # 2. Contour Prior
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. Decision: Use largest contour + saliency to seed GrabCut
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            rect = cv2.boundingRect(c)
            # Expand rect slightly for GrabCut
            x, y, rw, rh = rect
            pad_w = int(rw * 0.1)
            pad_h = int(rh * 0.1)
            rect_expanded = (max(0, x-pad_w), max(0, y-pad_h), min(w, rw+2*pad_w), min(h, rh+2*pad_h))
            
            # Use GrabCut with mask initialized by saliency if confidence is high, 
            # else use rect initialization.
            cv2.grabCut(frame, mask, rect_expanded, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        else:
            # Fallback to center prior
            margin_h = int(h * 0.15)
            margin_w = int(w * 0.15)
            rect = (margin_w, margin_h, w - 2 * margin_w, h - 2 * margin_h)
            cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # Post-process mask
        binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Refine mask (Morphology)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate Rich Metrics
        pixel_count = np.sum(binary_mask)
        occupancy = float(pixel_count / (h * w))
        
        # Contour-based metrics
        mask_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fragment_count = len(mask_contours)
        
        solidity = 0.0
        largest_contour_ratio = 0.0
        bbox = (0, 0, 0, 0)
        centroid = (w//2, h//2)
        
        if mask_contours:
            main_c = max(mask_contours, key=cv2.contourArea)
            area = cv2.contourArea(main_c)
            hull = cv2.convexHull(main_c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area / hull_area) if hull_area > 0 else 0
            largest_contour_ratio = float(area / pixel_count) if pixel_count > 0 else 0
            bbox = cv2.boundingRect(main_c)
            
            M = cv2.moments(main_c)
            if M["m00"] != 0:
                centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Confidence heuristic
        confidence = 1.0
        if occupancy < 0.03 or occupancy > 0.95: confidence *= 0.4
        if solidity < 0.5: confidence *= 0.7
        if largest_contour_ratio < 0.8: confidence *= 0.7 # High fragmentation
        
        edge_pixels = np.sum(binary_mask[0, :]) + np.sum(binary_mask[-1, :]) + \
                      np.sum(binary_mask[:, 0]) + np.sum(binary_mask[:, -1])
        if edge_pixels > 0:
            confidence *= 0.6
            
        return binary_mask * 255, {
            "confidence": confidence,
            "occupancy": occupancy,
            "is_clipped": edge_pixels > 0,
            "bbox": bbox,
            "centroid": centroid,
            "fragment_count": fragment_count,
            "solidity": solidity,
            "largest_contour_ratio": largest_contour_ratio
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
