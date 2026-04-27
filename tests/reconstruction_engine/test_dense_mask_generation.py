import unittest
import os
import shutil
import numpy as np
import cv2
from pathlib import Path
from modules.reconstruction_engine.adapter import COLMAPAdapter

class TestDenseMaskGeneration(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_dense_mask_gen")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.dense_dir = self.test_dir / "dense"
        self.images_dir = self.dense_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.feature_masks_dir = self.test_dir / "masks"
        self.feature_masks_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.test_dir / "test.log", "w")

    def tearDown(self):
        self.log_file.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_dummy_image(self, path, shape=(100, 100, 3)):
        img = np.zeros(shape, dtype=np.uint8)
        cv2.imwrite(str(path), img)

    def create_dummy_mask(self, path, shape=(100, 100)):
        mask = np.zeros(shape, dtype=np.uint8)
        # Create a circle in the middle to pass occupancy check
        cv2.circle(mask, (shape[1]//2, shape[0]//2), shape[0]//4, 255, -1)
        cv2.imwrite(str(path), mask)

    def test_successful_mask_generation(self):
        # Create 3 images and 3 corresponding masks
        for i in range(3):
            self.create_dummy_image(self.images_dir / f"frame_{i:04d}.jpg")
            self.create_dummy_mask(self.feature_masks_dir / f"frame_{i:04d}.jpg.png")

        adapter = COLMAPAdapter()
        stats = adapter._generate_dense_masks_from_feature_masks(
            self.dense_dir, self.feature_masks_dir, self.log_file
        )

        self.assertEqual(stats["dense_image_count"], 3)
        self.assertEqual(stats["dense_mask_count"], 3)
        self.assertEqual(stats["dense_mask_exact_filename_matches"], 3)
        self.assertEqual(stats["dense_mask_fallback_white_count"], 0)
        self.assertEqual(stats["quality_status"], "pass")

        # Verify output files exist
        stereo_masks_dir = self.dense_dir / "stereo" / "masks"
        for i in range(3):
            self.assertTrue((stereo_masks_dir / f"frame_{i:04d}.jpg.png").exists())

    def test_white_fallback_on_missing_mask(self):
        self.create_dummy_image(self.images_dir / "frame_0001.jpg")
        # No mask created

        adapter = COLMAPAdapter()
        stats = adapter._generate_dense_masks_from_feature_masks(
            self.dense_dir, self.feature_masks_dir, self.log_file
        )

        self.assertEqual(stats["dense_mask_fallback_white_count"], 1)
        self.assertEqual(stats["quality_status"], "failed") # 1/1 = 100% fallback

    def test_white_fallback_on_occupancy_failure(self):
        self.create_dummy_image(self.images_dir / "frame_0001.jpg")
        # Create an almost empty mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255 
        cv2.imwrite(str(self.feature_masks_dir / "frame_0001.jpg.png"), mask)

        adapter = COLMAPAdapter()
        stats = adapter._generate_dense_masks_from_feature_masks(
            self.dense_dir, self.feature_masks_dir, self.log_file
        )

        self.assertEqual(stats["dense_mask_fallback_white_count"], 1)
        self.assertEqual(stats["quality_status"], "failed")

if __name__ == "__main__":
    unittest.main()
