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
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
from modules.qa_validation.validator import AssetValidator
from modules.qa_validation.rules import ValidationThresholds
from modules.export_pipeline.glb_exporter import GLBExporter
from modules.integration_flow import IntegrationFlow

class TestPart4HardenedFinal(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_part4_final")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.validator = AssetValidator()

    def test_raw_archive_strategy(self):
        """
        Verifies that raw_archive is explicitly not mobile-ready and marked archive_only.
        """
        report = self.validator.validate("asset_archive", {
            "poly_count": 500000,
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
        
        self.assertEqual(report.delivery_status, "archive_only")
        self.assertFalse(report.is_mobile_ready)
        self.assertEqual(report.final_decision, "pass")

    def test_mobile_delivery_ready(self):
        """
        Verifies that mobile profiles result in delivery_ready and mobile_ready=True.
        """
        report = self.validator.validate("asset_mobile", {
            "poly_count": 40000,
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
        
        self.assertEqual(report.delivery_status, "delivery_ready")
        self.assertTrue(report.is_mobile_ready)
        self.assertEqual(report.final_decision, "pass")

    def test_glb_exporter_delivery_ready_fail(self):
        """
        Verifies GLBExporter returns delivery_ready=False when strict inspection fails.
        """
        exporter = GLBExporter()
        mesh = trimesh.creation.box()
        glb_path = self.test_dir / "failed_delivery.glb"
        mesh.export(str(glb_path))
        
        # Mock inspection failure (e.g. missing normals)
        with patch.object(exporter, "inspect_exported_asset") as mock_inspect:
            mock_inspect.return_value = {
                "all_primitives_have_position": True,
                "all_primitives_have_normal": False, # FAIL
                "all_textured_primitives_have_texcoord_0": True,
                "primitive_attribute_report": [],
                "texture_count": 0,
                "material_count": 0
            }
            
            result = exporter.export(str(glb_path), str(self.test_dir / "out.glb"))
            self.assertFalse(result["delivery_ready"])
            self.assertEqual(result["export_status"], "failed_validation")

    def test_integration_flow_bridging(self):
        """
        Verifies IntegrationFlow merges real cleanup, export and texture metrics.
        """
        metadata = NormalizedMetadata(
            bbox_min={"x":0,"y":0,"z":0},
            bbox_max={"x":10,"y":10,"z":10},
            pivot_offset={"z":0},
            final_polycount=50000
        )
        
        cleanup_stats = {
            "has_uv": True,
            "decimation": {"decimation_status": "success", "uv_preserved": True}
        }
        
        export_report = {
            "profile": "mobile_high",
            "delivery_ready": True,
            "all_primitives_have_position": True,
            "all_primitives_have_normal": True,
            "all_textured_primitives_have_texcoord_0": True
        }
        
        validator_input = IntegrationFlow.map_metadata_to_validator_input(
            metadata, 
            cleanup_stats=cleanup_stats,
            export_report=export_report,
            texture_quality_status="success",
            filtering_status="object_isolated"
        )
        
        report = self.validator.validate("integration_test", validator_input, allow_texture_quality_skip=True)
        print(f"\nINTEGRATION REPORT: decision={report.final_decision}, blocking={report.blocking_checks}, warnings={report.warning_checks}")
        
        self.assertEqual(report.delivery_status, "delivery_ready")
        self.assertTrue(report.is_mobile_ready)
        self.assertIn("decimation", report.contamination_report)

if __name__ == "__main__":
    unittest.main()
