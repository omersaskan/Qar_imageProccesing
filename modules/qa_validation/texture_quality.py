import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

class TextureQualityAnalyzer:
    def __init__(self, thresholds=None):
        from modules.operations.settings import settings
        self.settings = settings
        # Ensure we have thresholds for the new hard gates
        self.thresholds = thresholds or settings

    def analyze_path(self, image_path: str, expected_product_color: str = "unknown") -> Dict[str, Any]:
        """
        Loads image from path and analyzes its quality.
        """
        if not image_path or not Path(image_path).exists():
            return self._error_result(f"Texture file missing: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
             try:
                 img_bytes = np.fromfile(image_path, dtype=np.uint8)
                 image = cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
             except Exception:
                 pass
                 
        if image is None:
            return self._error_result(f"Could not read image: {image_path}")
        return self.analyze_image(image, expected_product_color)

    def analyze_image(self, image: np.ndarray, expected_product_color: str = "unknown") -> Dict[str, Any]:
        """
        Core analysis logic using NumPy and OpenCV.
        """
        # 1. Handle Alpha / Transparency
        has_alpha = image.shape[2] == 4 if len(image.shape) == 3 else False
        if has_alpha:
            alpha = image[:, :, 3]
            mask = alpha > 0
            rgb = image[:, :, :3]
        else:
            if len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                mask = np.ones(image.shape, dtype=bool)
            else:
                mask = np.ones(image.shape[:2], dtype=bool)
                rgb = image

        visible_pixels = np.count_nonzero(mask)
        total_pixels = image.shape[0] * image.shape[1]
        
        if visible_pixels == 0:
            return self._error_result("Texture atlas is completely transparent or empty")

        # 2. Metric Computation
        max_rgb = np.max(rgb, axis=2)
        
        # A) Black Pixel Ratio (max(RGB) < 25)
        black_pixels = np.count_nonzero((max_rgb < 25) & mask)
        black_pixel_ratio = black_pixels / visible_pixels

        # B) Near Black Ratio (max(RGB) < 45)
        near_black_pixels = np.count_nonzero((max_rgb < 45) & mask)
        near_black_ratio = near_black_pixels / visible_pixels

        # C) Near White Ratio (HSV: High Value, Low Saturation)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        white_pixels = np.count_nonzero((v > 180) & (s < 60) & mask)
        near_white_ratio = white_pixels / visible_pixels

        # D) Default Fill or Flat Color Ratio
        # Neutral flat areas: low saturation, non-extreme value
        flat_color_pixels = np.count_nonzero((s < 12) & (v > 35) & (v < 220) & mask)
        default_fill_ratio = flat_color_pixels / visible_pixels
        
        # E) Dominant Color Ratio (most frequent hue bin)
        hue_hist = cv2.calcHist([h], [0], mask.astype(np.uint8), [180], [0, 180])
        dominant_hue_count = np.max(hue_hist)
        dominant_color_ratio = dominant_hue_count / visible_pixels

        # F) Dominant Background (Specific warm leakage)
        # Warm/Orange: Hue between 5 and 25
        h_visible = h[mask]
        s_visible = s[mask]
        v_visible = v[mask]
        orange_pixels = np.count_nonzero((h_visible >= 5) & (h_visible <= 25) & (s_visible > 50) & (v_visible > 50))
        orange_ratio = orange_pixels / visible_pixels
        
        # Generic non-neutral ratio
        non_neutral_mask = (s_visible > 40) & (v_visible > 40) & (v_visible < 240)
        non_neutral_count = np.count_nonzero(non_neutral_mask)
        non_neutral_ratio = non_neutral_count / visible_pixels
        
        dominant_background_ratio = orange_ratio
        if expected_product_color != "colorful":
            dominant_background_ratio = max(orange_ratio, non_neutral_ratio * 0.7)

        # G) Atlas Coverage
        atlas_coverage_ratio = visible_pixels / total_pixels

        # H) Alpha Empty Ratio
        alpha_empty_ratio = np.count_nonzero(alpha == 0) / total_pixels if has_alpha else 0.0

        # 3. Decision Logic (Hard Gates)
        reasons = []
        status = "success"
        
        # Thresholds (using safe defaults if settings missing)
        thr_black = getattr(self.thresholds, "max_black_pixel_ratio", 0.4)
        thr_near_black = getattr(self.thresholds, "max_near_black_ratio", 0.6)
        thr_flat = getattr(self.thresholds, "max_flat_color_ratio", 0.7)
        thr_coverage = getattr(self.thresholds, "min_atlas_coverage_ratio", 0.05)
        thr_bg = getattr(self.thresholds, "max_dominant_background_ratio", 0.5)

        if black_pixel_ratio > thr_black and expected_product_color != "dark":
            reasons.append(f"High black pixel ratio: {black_pixel_ratio:.2f} > {thr_black}")
            status = "fail"
        
        if near_black_ratio > thr_near_black and expected_product_color != "dark":
            reasons.append(f"High near-black ratio: {near_black_ratio:.2f} > {thr_near_black}")
            status = "fail"

        if default_fill_ratio > thr_flat:
            reasons.append(f"High flat color / default fill ratio: {default_fill_ratio:.2f} > {thr_flat}")
            status = "fail"

        if atlas_coverage_ratio < thr_coverage:
            reasons.append(f"Low atlas coverage: {atlas_coverage_ratio:.2f} < {thr_coverage}")
            status = "review"

        if dominant_background_ratio > thr_bg:
            reasons.append(f"Excessive background color contamination: {dominant_background_ratio:.2f} > {thr_bg}")
            status = "fail"

        return {
            "black_pixel_ratio": float(black_pixel_ratio),
            "near_black_ratio": float(near_black_ratio),
            "near_white_ratio": float(near_white_ratio),
            "dominant_color_ratio": float(dominant_color_ratio),
            "dominant_background_color_ratio": float(dominant_background_ratio),
            "atlas_coverage_ratio": float(atlas_coverage_ratio),
            "default_fill_or_flat_color_ratio": float(default_fill_ratio),
            "alpha_empty_ratio": float(alpha_empty_ratio),
            "texture_quality_status": status,
            "texture_quality_reasons": reasons
        }

    def _error_result(self, message: str) -> Dict[str, Any]:
        return {
            "texture_quality_status": "fail",
            "texture_quality_reasons": [f"ERROR: {message}"],
            "black_pixel_ratio": 0.0,
            "near_black_ratio": 0.0,
            "near_white_ratio": 0.0,
            "dominant_color_ratio": 0.0,
            "dominant_background_color_ratio": 0.0,
            "atlas_coverage_ratio": 0.0,
            "default_fill_or_flat_color_ratio": 0.0,
            "alpha_empty_ratio": 0.0
        }
