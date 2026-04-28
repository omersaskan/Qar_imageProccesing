import unittest
import os
import shutil
import numpy as np
import cv2
from pathlib import Path
from modules.asset_cleanup_pipeline.camera_projection import (
    load_reconstruction_cameras,
    load_reconstruction_masks,
    read_next_bytes
)
from modules.asset_cleanup_pipeline.isolation import MeshIsolator
import trimesh

class TestLegacyRootCauseFixes(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("scratch/test_root_cause_workspace")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)
        
        # Create a mock attempt_dir structure
        self.attempt_dir = self.test_dir / "attempt_1"
        self.attempt_dir.mkdir()
        
        self.dense_dir = self.attempt_dir / "dense"
        self.dense_dir.mkdir()
        
        self.original_sparse = self.attempt_dir / "sparse" / "0"
        self.original_sparse.mkdir(parents=True)
        
        self.undistorted_sparse = self.dense_dir / "sparse"
        self.undistorted_sparse.mkdir(parents=True)
        
        # Helper to write mock COLMAP binary files
        def write_mock_colmap_bin(path, is_camera=True, width=100, height=200, model_id=1):
            with open(path, "wb") as f:
                if is_camera:
                    f.write((1).to_bytes(8, 'little')) # num_cameras
                    # camera_id (i), model_id (i), width (Q), height (Q)
                    f.write((1).to_bytes(4, 'little'))
                    f.write((model_id).to_bytes(4, 'little'))
                    f.write((width).to_bytes(8, 'little'))
                    f.write((height).to_bytes(8, 'little'))
                    # params (d*4 for PINHOLE)
                    f.write(np.array([1000.0, 1000.0, 50.0, 100.0], dtype=np.float64).tobytes())
                else:
                    # images.bin
                    f.write((1).to_bytes(8, 'little')) # num_images
                    # image_id (i), qvec (4d), tvec (3d), camera_id (i)
                    f.write((1).to_bytes(4, 'little'))
                    f.write(np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64).tobytes())
                    f.write((1).to_bytes(4, 'little'))
                    # name (string + \0)
                    f.write(b"frame_0001.jpg\x00")
                    # num_points2D (Q)
                    f.write((0).to_bytes(8, 'little'))

        write_mock_colmap_bin(self.original_sparse / "cameras.bin", width=3840, height=2160, model_id=3) # RADIAL
        write_mock_colmap_bin(self.original_sparse / "images.bin", is_camera=False)
        
        write_mock_colmap_bin(self.undistorted_sparse / "cameras.bin", width=2000, height=1125, model_id=1) # PINHOLE
        write_mock_colmap_bin(self.undistorted_sparse / "images.bin", is_camera=False)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_camera_selection_prefers_undistorted(self):
        # Test loading from attempt_dir
        cameras = load_reconstruction_cameras(self.attempt_dir)
        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0]["width"], 2000)
        self.assertEqual(cameras[0]["camera_model_space"], "undistorted_dense")
        
        # Test loading from dense_dir
        cameras_dense = load_reconstruction_cameras(self.dense_dir)
        self.assertEqual(len(cameras_dense), 1)
        self.assertEqual(cameras_dense[0]["width"], 2000)
        self.assertEqual(cameras_dense[0]["camera_model_space"], "undistorted_dense")

    def test_mask_loading_paths_and_naming(self):
        # Create masks in dense/stereo/masks
        mask_dir = self.dense_dir / "stereo" / "masks"
        mask_dir.mkdir(parents=True)
        
        # One with .jpg.png, one with .png
        cv2.imwrite(str(mask_dir / "frame_0001.jpg.png"), np.zeros((10, 10), dtype=np.uint8))
        cv2.imwrite(str(mask_dir / "frame_0002.png"), np.zeros((10, 10), dtype=np.uint8))
        
        # Test loading from attempt_dir
        masks = load_reconstruction_masks(self.attempt_dir, ["frame_0001.jpg", "frame_0002.jpg"])
        self.assertIn("frame_0001.jpg", masks)
        self.assertIn("frame_0002.jpg", masks)
        self.assertEqual(len(masks), 2)
        
        # Test loading from dense_dir
        masks_dense = load_reconstruction_masks(self.dense_dir, ["frame_0001.jpg", "frame_0002.jpg"])
        self.assertIn("frame_0001.jpg", masks_dense)
        self.assertEqual(len(masks_dense), 2)

    def test_mask_auto_resize(self):
        mask_dir = self.dense_dir / "stereo" / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Write a 100x100 mask
        cv2.imwrite(str(mask_dir / "frame_0001.jpg.png"), np.zeros((100, 100), dtype=np.uint8))
        
        # Load with expected 200x112
        masks = load_reconstruction_masks(self.attempt_dir, ["frame_0001.jpg"], expected_width=200, expected_height=112)
        self.assertEqual(masks["frame_0001.jpg"].shape, (112, 200))

    def test_isolation_quality_metrics(self):
        isolator = MeshIsolator()
        
        # Mock components: one tiny (52 faces), one larger (1000 faces)
        # Use structured meshes to avoid connected_components bottlenecks
        c1 = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        # subdivide to get more faces
        c1 = c1.subdivide().subdivide().subdivide() # 6 * 4^3 * 2 = 768 faces roughly
        
        c2 = trimesh.creation.box(extents=(1, 1, 1))
        
        mesh = trimesh.util.concatenate([c1, c2])
        mesh.metadata['face_indices'] = np.arange(len(mesh.faces))
        
        # Run isolation (geometric only for mock)
        isolated, stats = isolator.isolate_product(mesh)
        
        self.assertIn("primary_face_share", stats)
        self.assertIn("largest_kept_component_share", stats)
        self.assertIn("kept_to_initial_face_ratio", stats)
        self.assertGreater(stats["primary_face_share"], 0)

if __name__ == "__main__":
    unittest.main()
