import unittest
import os
import json
import shutil
import numpy as np
import trimesh
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure we can import modules from the parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent))

from modules.reconstruction_engine.output_manifest import OutputManifest
from modules.reconstruction_engine.texture_frame_filter import TextureFrameFilter
from modules.operations.texturing_service import TexturingService
from modules.asset_cleanup_pipeline.remesher import Remesher
from modules.asset_cleanup_pipeline.profiles import PROFILES, CleanupProfileType
from modules.integration_flow import IntegrationFlow
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata

class TestSprint5CFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)
        
    def setUp(self):
        self.test_dir = Path("tests/scratch_sprint5c")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # We keep scratch for debugging if requested, but normally we clean up
        # However, the user wants the test to stay in the repo.
        if self.test_dir.exists():
             shutil.rmtree(self.test_dir)

    def test_mask_resolution(self):
        """
        Verify that TextureFrameFilter finds masks in various deep paths.
        """
        image_dir = self.test_dir / "dense" / "images"
        image_dir.mkdir(parents=True)
        
        # Scenario: masks are in dense/stereo/masks
        mask_dir = self.test_dir / "dense" / "stereo" / "masks"
        mask_dir.mkdir(parents=True)
        
        # Create a dummy image and mask
        import cv2
        img_path = image_dir / "frame_0001.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        
        mask_path = mask_dir / "frame_0001.png"
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.imwrite(str(mask_path), mask)
        
        filter = TextureFrameFilter()
        results = filter.filter_session_images(image_dir, self.test_dir)
        
        self.assertTrue(results["has_masks_available"])
        # Check if masked image was generated
        masked_img_path = Path(results["masked_images_dir"]) / "frame_0001.jpg"
        self.assertTrue(masked_img_path.exists())

    def test_mobile_high_polycount_reduction(self):
        """
        Verify that mobile_high 190k face textured OBJ is reduced below 150k.
        """
        mesh_path = self.test_dir / "high_poly.obj"
        # Create a mesh with ~600k faces to ensure it triggers fallback if needed
        mesh = trimesh.creation.uv_sphere(radius=1.0, count=[400, 400])
        # Ensure it has UVs
        mesh.visual = trimesh.visual.TextureVisuals(uv=np.random.rand(len(mesh.vertices), 2))
        mesh.export(str(mesh_path))
        
        pre_faces = len(mesh.faces)
        self.assertGreater(pre_faces, 150000)
        
        remesher = Remesher()
        output_path = self.test_dir / "cleaned_mesh.obj"
        profile = PROFILES[CleanupProfileType.MOBILE_HIGH]
        
        stats = remesher.process(str(mesh_path), str(output_path), profile)
        
        post_mesh = trimesh.load(str(output_path))
        post_faces = len(post_mesh.faces)
        
        print(f"Pre-faces: {pre_faces}, Post-faces: {post_faces}")
        self.assertLessEqual(post_faces, profile.face_count_limit)
        self.assertIn(stats["decimation_status"], ["success", "success_fallback"])

    @patch("modules.reconstruction_engine.openmvs_texturer.OpenMVSTexturer")
    @patch("modules.operations.atlas_repair_service.AtlasRepairService")
    def test_texturing_retry_path_update(self, mock_repair, mock_texturer):
        """
        Verify that retry replaces textured_path and final aligned OBJ comes from retry.
        """
        # Mock initial run: high contamination
        mock_repair_instance = mock_repair.return_value
        mock_repair_instance.repair_atlas.side_effect = [
            {"status": "fail", "stats": {"dominant_background_color_ratio": 0.8}, "repaired_path": None}, # Initial
            {"status": "success", "stats": {"dominant_background_color_ratio": 0.1}, "repaired_path": "retry_path.png"} # Retry
        ]
        
        mock_texturer_instance = mock_texturer.return_value
        mock_texturer_instance.run_texturing.side_effect = [
            {"textured_mesh_path": "original.obj", "texture_atlas_paths": ["original.png"]},
            {"textured_mesh_path": "retry.obj", "texture_atlas_paths": ["retry.png"]}
        ]
        
        service = TexturingService()
        
        cleanup_stats = {
            "pre_aligned_mesh_path": "pre_aligned.obj",
            "metadata_path": "meta.json",
            "pivot_offset": {"x": 0, "y": 0, "z": 0}
        }
        
        # Create dummy files and structure to pass checks
        Path("pre_aligned.obj").touch()
        Path("meta.json").write_text("{}")
        Path("dense").mkdir(exist_ok=True)
        
        pivot_offset = {"x": 0, "y": 0, "z": 0}
        cleaned_mesh_path = "final_cleaned.obj"
        
        # Mock manifest
        manifest = MagicMock(spec=OutputManifest)
        manifest.mesh_path = "dense/raw.obj"
        manifest.engine_type = "openmvs"
        manifest.mesh_metadata = MagicMock()
        
        with patch("modules.operations.texturing_service.trimesh.load") as mock_trimesh_load:
            # Create a real mesh for the mock to return
            mock_mesh = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]], faces=[[0,1,2]])
            mock_mesh.visual = trimesh.visual.TextureVisuals(uv=[[0,0],[1,0],[0,1]])
            mock_trimesh_load.return_value = mock_mesh
            
            with patch("modules.operations.texturing_service.atomic_write_json"):
                with patch("shutil.copy2"):
                     # Mock _apply_pivot_to_obj to avoid file IO errors
                     service._apply_pivot_to_obj = MagicMock(return_value="final.obj")
                     
                     results = service.run(manifest, cleanup_stats, pivot_offset, cleaned_mesh_path, expected_color="white_cream")
                     
                     # Verify that retry was called
                     self.assertEqual(mock_texturer_instance.run_texturing.call_count, 2)
                     # Verify that final output came from retry results
                     service._apply_pivot_to_obj.assert_called_with("retry.obj", unittest.mock.ANY, unittest.mock.ANY)

        # Cleanup dummy files
        Path("pre_aligned.obj").unlink()
        Path("meta.json").unlink()
        if Path("dense").exists():
            shutil.rmtree("dense")

    def test_validation_consumption_of_report(self):
        """
        Verify validation_report consumes texture_quality_report and does not emit MISSING_TEXTURE_QUALITY_METRICS.
        """
        report_path = self.test_dir / "texture_quality_report.json"
        report_data = {
            "status": "success",
            "dominant_background_color_ratio": 0.05,
            "black_pixel_ratio": 0.1,
            "atlas_coverage_ratio": 0.8
        }
        with open(report_path, "w") as f:
            json.dump(report_data, f)
            
        metadata = NormalizedMetadata(
            final_polycount=1000,
            bbox_min={"x": 0, "y": 0, "z": 0},
            bbox_max={"x": 1, "y": 1, "z": 1},
            pivot_offset={"x": 0, "y": 0, "z": 0}
        )
        
        cleanup_stats = {
            "texture_quality_report_path": str(report_path),
            "has_uv": True,
            "has_material": True,
            "texture_integrity_status": "complete"
        }
        
        input_data = IntegrationFlow.map_metadata_to_validator_input(metadata, cleanup_stats=cleanup_stats)
        
        self.assertEqual(input_data["dominant_background_color_ratio"], 0.05)
        self.assertEqual(input_data["texture_quality_status"], "success")
        
        from modules.qa_validation.validator import AssetValidator
        validator = AssetValidator()
        report = validator.validate("test_asset", input_data)
        
        self.assertNotIn("MISSING_TEXTURE_QUALITY_METRICS", report.texture_quality_reasons)
        self.assertEqual(report.texture_quality_status, "success")

if __name__ == "__main__":
    unittest.main()
