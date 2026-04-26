import unittest
import cv2
import numpy as np
import os
from pathlib import Path
from modules.qa_validation.texture_quality import TextureQualityAnalyzer
from modules.qa_validation.rules import ValidationThresholds

class TestTextureQualityRealImages(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_texture_qa")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds = ValidationThresholds()
        # Tighten thresholds for test as per request
        self.thresholds.min_atlas_coverage_ratio = 0.30
        self.thresholds.max_black_pixel_ratio = 0.40
        self.thresholds.max_dominant_background_ratio = 0.50
        
        self.analyzer = TextureQualityAnalyzer(thresholds=self.thresholds)

    def _create_atlas(self, name: str, size: int = 512, bg_color=(0, 0, 0, 0), content_color=(200, 200, 200, 255), content_rect=None):
        path = self.test_dir / f"{name}.png"
        # Create BG
        img = np.zeros((size, size, 4), dtype=np.uint8)
        img[:] = bg_color
        
        if content_rect:
            x, y, w, h = content_rect
            img[y:y+h, x:x+w] = content_color
            
        cv2.imwrite(str(path), img)
        return str(path)

    def test_mostly_black_atlas_fails(self):
        # 80% black content, 20% transparent (visible is all black)
        path = self._create_atlas("mostly_black", bg_color=(0,0,0,0), content_color=(5,5,5,255), content_rect=(0,0,512,512))
        results = self.analyzer.analyze_path(path)
        self.assertEqual(results["texture_quality_status"], "fail")
        self.assertTrue(any("black" in r.lower() for r in results["texture_quality_reasons"]))

    def test_orange_contaminated_atlas_fails(self):
        # 60% orange content (leakage from background)
        # BGR: Orange is roughly (0, 165, 255)
        path = self._create_atlas("orange_contaminated", content_color=(0, 150, 250, 255), content_rect=(0,0,512,512))
        results = self.analyzer.analyze_path(path)
        self.assertEqual(results["texture_quality_status"], "fail")
        self.assertTrue(any("background color contamination" in r.lower() for r in results["texture_quality_reasons"]))

    def test_low_coverage_atlas_fails(self):
        # 10% coverage (Threshold is 0.30)
        path = self._create_atlas("low_coverage", content_color=(200,200,200,255), content_rect=(0,0,100,100))
        results = self.analyzer.analyze_path(path)
        # Low coverage is currently marked as 'review' in texture_quality.py
        self.assertEqual(results["texture_quality_status"], "review")
        self.assertTrue(any("low atlas coverage" in r.lower() for r in results["texture_quality_reasons"]))

    def test_clean_atlas_passes(self):
        # Create a "clean" but non-flat atlas using noise
        path = self.test_dir / "clean_atlas.png"
        img = np.zeros((512, 512, 4), dtype=np.uint8)
        
        # Content with colorful noise to avoid "flat color" (s < 12)
        # We'll use Blue/Green tones to avoid "orange" (h: 5-25)
        img_hsv = np.zeros((400, 400, 3), dtype=np.uint8)
        img_hsv[:, :, 0] = np.random.randint(100, 140, (400, 400), dtype=np.uint8) # Hue (Blue/Green)
        img_hsv[:, :, 1] = np.random.randint(100, 200, (400, 400), dtype=np.uint8) # Saturation (Avoid flat)
        img_hsv[:, :, 2] = np.random.randint(100, 200, (400, 400), dtype=np.uint8) # Value
        
        img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        img[0:400, 0:400, :3] = img_bgr
        img[0:400, 0:400, 3] = 255 # Alpha
        cv2.imwrite(str(path), img)

        results = self.analyzer.analyze_path(str(path), expected_product_color="colorful")
        print(f"\nCLEAN ATLAS RESULTS: {results}")
        self.assertEqual(results["texture_quality_status"], "success")

if __name__ == "__main__":
    unittest.main()
