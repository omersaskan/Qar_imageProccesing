import unittest
import os
import json
import trimesh
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock

from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType
from modules.qa_validation.validator import AssetValidator
from modules.qa_validation.rules import ValidationThresholds
from modules.export_pipeline.glb_exporter import GLBExporter

class TestPart4HardenedVerification(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_part4")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy texture
        self.dummy_tex = self.test_dir / "dummy_tex.png"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:80, 20:80] = [200, 200, 200] # Some content
        cv2.imwrite(str(self.dummy_tex), img)

    def test_mobile_preview_face_count_limit(self):
        """
        Verifies that mobile_preview profile fails if polycount exceeds 50k.
        """
        validator = AssetValidator()
        
        # 1. 45k faces (Pass/Review)
        report_45k = validator.validate("asset_45k", {
            "poly_count": 45000,
            "delivery_profile": "mobile_preview",
            "all_primitives_have_position": True,
            "all_primitives_have_normal": True,
            "all_textured_primitives_have_texcoord_0": True,
            "material_semantic_status": "diffuse_textured",
            "texture_integrity_status": "complete",
            "texture_quality_status": "success",
            "filtering_status": "object_isolated",
            "bbox": {"width": 10, "height": 10, "depth": 10},
            "ground_offset": 0.0
        }, allow_texture_quality_skip=True)
        
        # 2. 55k faces (Fail)
        report_55k = validator.validate("asset_55k", {
            "poly_count": 55000,
            "delivery_profile": "mobile_preview",
            "all_primitives_have_position": True,
            "all_primitives_have_normal": True,
            "all_textured_primitives_have_texcoord_0": True,
            "material_semantic_status": "diffuse_textured",
            "texture_integrity_status": "complete",
            "texture_quality_status": "success",
            "filtering_status": "object_isolated",
            "bbox": {"width": 10, "height": 10, "depth": 10},
            "ground_offset": 0.0
        }, allow_texture_quality_skip=True)
        
        print(f"\nDEBUG 45k: decision={report_45k.final_decision}, blocking={report_45k.blocking_checks}")
        self.assertIn(report_45k.final_decision, ["pass", "review"])
        self.assertEqual(report_55k.final_decision, "fail")
        self.assertIn("polycount", report_55k.blocking_checks)

    def test_raw_archive_preservation(self):
        """
        Verifies that raw_archive can preserve high polycount but is not mobile ready.
        """
        validator = AssetValidator()
        report = validator.validate("asset_hq", {
            "poly_count": 1000000,
            "delivery_profile": "raw_archive",
            "all_primitives_have_position": True,
            "all_primitives_have_normal": True,
            "all_textured_primitives_have_texcoord_0": True,
            "material_semantic_status": "diffuse_textured",
            "texture_integrity_status": "complete",
            "texture_quality_status": "success",
            "filtering_status": "object_isolated",
            "bbox": {"width": 10, "height": 10, "depth": 10},
            "ground_offset": 0.0
        }, allow_texture_quality_skip=True)
        
        self.assertEqual(report.final_decision, "pass")
        self.assertEqual(report.mobile_performance_grade, "D")

    def test_textured_decimation_integrity(self):
        """
        Verifies that textured decimation fails if it destroys UVs.
        """
        from modules.asset_cleanup_pipeline.remesher import Remesher
        from modules.asset_cleanup_pipeline.profiles import PROFILES
        
        remesher = Remesher()
        profile = PROFILES[CleanupProfileType.MOBILE_PREVIEW]
        
        # Mock a mesh with UVs using a clean primitive (box)
        mesh = trimesh.creation.box()
        # Subdivide many times to exceed 45k target (12 * 4^8 = 786k)
        for _ in range(8):
            mesh = mesh.subdivide()
        mesh.visual = trimesh.visual.TextureVisuals(uv=np.random.rand(len(mesh.vertices), 2))
        
        with patch("trimesh.load", return_value=mesh), \
             patch.object(trimesh.Trimesh, "simplify_quadric_decimation") as mock_simp, \
             patch.object(trimesh.Trimesh, "export"):
            
            # Scenario: simplification removes UVs
            bad_candidate = trimesh.Trimesh(vertices=np.random.rand(100, 3), faces=np.random.randint(0, 100, (50, 3)))
            mock_simp.return_value = bad_candidate
            
            stats = remesher.process("dummy.obj", "output.obj", profile)
            
            self.assertIn(stats["decimation_status"], ["failed_visual_integrity", "success_fallback_destructive"])
            self.assertFalse(stats["uv_preserved"])

    def test_final_glb_strict_accessor_inspection(self):
        """
        Verifies that GLBExporter performs strict inspection and detects missing NORMALs.
        """
        exporter = GLBExporter()
        
        # Create a clean box mesh and export it
        mesh = trimesh.creation.box()
        glb_path = self.test_dir / "test_accessors.glb"
        mesh.export(str(glb_path))
        
        # 1. Inspect valid
        inspection = exporter.inspect_exported_asset(str(glb_path))
        self.assertTrue(inspection["all_primitives_have_position"])
        
        # 2. Test the strict JSON parser logic by injecting a fake GLB without NORMAL
        with patch("modules.export_pipeline.glb_exporter.inspect_glb_primitive_attributes") as mock_strict:
            mock_strict.return_value = {
                "all_primitives_have_position": True,
                "all_primitives_have_normal": False,
                "all_textured_primitives_have_texcoord_0": True,
                "primitive_attribute_report": [],
                "texture_count": 0,
                "material_count": 0
            }
            
            inspection_fail = exporter.inspect_exported_asset(str(glb_path))
            self.assertFalse(inspection_fail["all_primitives_have_normal"])

    def test_texture_qa_hard_gate(self):
        """
        Verifies that contaminated texture atlas fails validation.
        """
        validator = AssetValidator()
        
        # 1. Clean
        report_clean = validator.validate("asset_clean", {
            "texture_quality_status": "success",
            "material_semantic_status": "diffuse_textured",
            "poly_count": 10000,
            "delivery_profile": "mobile_preview",
            "all_primitives_have_position": True,
            "all_primitives_have_normal": True,
            "all_textured_primitives_have_texcoord_0": True,
            "texture_integrity_status": "complete",
            "filtering_status": "object_isolated",
            "bbox": {"width": 10, "height": 10, "depth": 10},
            "ground_offset": 0.0
        }, allow_texture_quality_skip=True)
        
        # 2. Contaminated
        report_fail = validator.validate("asset_dirty", {
            "texture_quality_status": "fail",
            "material_semantic_status": "diffuse_textured",
            "poly_count": 10000,
            "delivery_profile": "mobile_preview",
            "all_primitives_have_position": True,
            "all_primitives_have_normal": True,
            "all_textured_primitives_have_texcoord_0": True,
            "texture_integrity_status": "complete",
            "filtering_status": "object_isolated",
            "bbox": {"width": 10, "height": 10, "depth": 10},
            "ground_offset": 0.0
        }, allow_texture_quality_skip=True)
        
        self.assertEqual(report_clean.final_decision, "pass")
        self.assertEqual(report_fail.final_decision, "fail")
        self.assertIn("texture_quality", report_fail.blocking_checks)

if __name__ == "__main__":
    unittest.main()
