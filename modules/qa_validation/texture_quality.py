import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

class TextureQualityAnalyzer:
    def __init__(self, thresholds=None):
        from modules.operations.settings import settings
        # Use provided thresholds or global settings
        self.settings = settings
        self.thresholds = thresholds or settings

    def analyze_path(self, image_path: str, expected_product_color: str = "unknown") -> Dict[str, Any]:
        """
        Loads image from path and analyzes its quality.
        """
        # Read with alpha channel support
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
             # Try reading with numpy/imdecode for unicode paths on Windows
             try:
                 img_bytes = np.fromfile(image_path, dtype=np.uint8)
                 image = cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
             except Exception:
                 pass
                 
        if image is None:
            return self._error_result(f"Could not read image: {image_path}")
        return self.analyze_image(image, expected_product_color)

    def analyze_bytes(self, image_bytes: bytes, expected_product_color: str = "unknown") -> Dict[str, Any]:
        """
        Decodes image bytes and analyzes its quality.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if image is None:
            return self._error_result("Could not decode image bytes")
        return self.analyze_image(image, expected_product_color)

    def analyze_image(self, image: np.ndarray, expected_product_color: str = "unknown") -> Dict[str, Any]:
        """
        Core analysis logic using NumPy and OpenCV.
        """
        # 1. Handle Alpha / Transparency
        # 4 channels: BGR + Alpha
        has_alpha = image.shape[2] == 4 if len(image.shape) == 3 else False
        if has_alpha:
            alpha = image[:, :, 3]
            # Consider only pixels that are at least partially opaque
            mask = alpha > 0
            rgb = image[:, :, :3]
        else:
            # If 2D (grayscale) or 3D without alpha
            if len(image.shape) == 2:
                # Convert grayscale to BGR for uniform processing
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                mask = np.ones(image.shape, dtype=bool)
            else:
                mask = np.ones(image.shape[:2], dtype=bool)
                rgb = image

        visible_pixels = np.count_nonzero(mask)
        if visible_pixels == 0:
            return self._error_result("Texture atlas is completely transparent or empty")

        # 2. Metric Computation
        
        # A) Black Pixel Ratio (max(RGB) < 25)
        # Using vectorized max across channels
        max_rgb = np.max(rgb, axis=2)
        black_pixels = np.count_nonzero((max_rgb < 25) & mask)
        black_pixel_ratio = black_pixels / visible_pixels

        # B) Near White Ratio (HSV: High Value, Low Saturation)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # white/cream: high value (>180), low saturation (<60)
        white_pixels = np.count_nonzero((v > 180) & (s < 60) & mask)
        near_white_ratio = white_pixels / visible_pixels

        # C) Dominant Background / Color Clusters
        h_visible = h[mask]
        s_visible = s[mask]
        v_visible = v[mask]
        
        # Warm/Orange (Common leakage): Hue between 5 and 25 (OpenCV Hue is 0-180)
        orange_pixels = np.count_nonzero((h_visible >= 5) & (h_visible <= 25) & (s_visible > 50) & (v_visible > 50))
        orange_ratio = orange_pixels / visible_pixels

        # Generic dominant non-neutral color detection
        # Neutral: low saturation or extreme value
        non_neutral_mask = (s_visible > 40) & (v_visible > 40) & (v_visible < 240)
        non_neutral_count = np.count_nonzero(non_neutral_mask)
        non_neutral_ratio = non_neutral_count / visible_pixels
        
        # Dominant background is either the specific orange leakage or high non-neutral saturation
        dominant_background_ratio = orange_ratio
        if expected_product_color != "colorful":
            dominant_background_ratio = max(orange_ratio, non_neutral_ratio * 0.7)
        else:
            # For colorful products, only specific common background hues like orange/warm are suspicious
            dominant_background_ratio = orange_ratio

        # D) Atlas Coverage
        total_pixels = image.shape[0] * image.shape[1]
        atlas_coverage_ratio = visible_pixels / total_pixels

        # 3. Decision Logic
        reasons = []
        status = "clean"
        
        # Black ratio check (skip or relax for dark products)
        if black_pixel_ratio > self.thresholds.max_black_pixel_ratio:
            if expected_product_color != "dark":
                reasons.append("BLACK_PATCH_RATIO_HIGH")
                status = "contaminated"
            elif black_pixel_ratio > 0.8: # Even for dark products, 80% black might be empty
                reasons.append("BLACK_PATCH_RATIO_EXTREME")
                status = "warning"
        
        # Background contamination check
        if dominant_background_ratio > self.thresholds.max_dominant_background_ratio:
            reasons.append("BACKGROUND_COLOR_CONTAMINATION")
            status = "contaminated"

        # Coverage check
        if atlas_coverage_ratio < self.thresholds.min_atlas_coverage_ratio:
            reasons.append("LOW_ATLAS_COVERAGE")
            if status == "clean": 
                status = "warning"

        # Product-specific: White/Cream check
        if expected_product_color == "white_cream":
            if near_white_ratio < self.thresholds.min_near_white_ratio_white_cream:
                reasons.append("WHITE_PRODUCT_TEXTURE_RATIO_LOW")
                status = "contaminated"

        # Grading (A-F)
        grade = "A"
        if status == "contaminated":
            grade = "F"
        elif status == "warning":
            grade = "C"
        elif black_pixel_ratio > 0.1 or dominant_background_ratio > 0.05:
            grade = "B"

        return {
            "black_pixel_ratio": float(black_pixel_ratio),
            "near_white_ratio": float(near_white_ratio),
            "dominant_background_color_ratio": float(dominant_background_ratio),
            "atlas_coverage_ratio": float(atlas_coverage_ratio),
            "texture_quality_status": status,
            "texture_quality_grade": grade,
            "texture_quality_reasons": reasons
        }

    def _error_result(self, message: str) -> Dict[str, Any]:
        return {
            "texture_quality_status": "invalid",
            "texture_quality_grade": "F",
            "texture_quality_reasons": [f"ERROR: {message}"],
            "black_pixel_ratio": 0.0,
            "near_white_ratio": 0.0,
            "dominant_background_color_ratio": 0.0,
            "atlas_coverage_ratio": 0.0
        }
