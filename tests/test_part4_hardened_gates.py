import unittest
import trimesh
import numpy as np
from pathlib import Path
from unittest.mock import patch

from modules.export_pipeline.glb_exporter import GLBExporter
from modules.qa_validation.validator import AssetValidator
from modules.integration_flow import IntegrationFlow
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata

class TestPart4HardenedGates(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_part4_hardened")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.validator = AssetValidator()

    def test_glb_exporter_texture_failure_gate(self):
        """
        Verifies that GLBExporter fails delivery_ready if texture_path is provided but application fails.
        """
        exporter = GLBExporter()
        mesh = trimesh.creation.box()
        # Ensure no UVs to trigger failure if applied
        mesh.visual = trimesh.visual.ColorVisuals() 
        
        glb_path = self.test_dir / "textured_fail.glb"
        mesh.export(str(glb_path))
        
        # Scenario: texture_path provided but mesh has no UV (mocking trimesh failure or our logic)
        with patch.object(exporter, "inspect_exported_asset") as mock_inspect:
            mock_inspect.return_value = {
                "all_primitives_have_position": True,
                "all_primitives_have_normal": True,
                "all_textured_primitives_have_texcoord_0": False, # NO UV
                "texture_count": 0,
                "material_count": 1
            }
            
            result = exporter.export(str(glb_path), str(self.test_dir / "out.glb"), texture_path="dummy.png")
            
            # Should be False because texture was requested but texture_count=0 or accessors failed
            self.assertFalse(result["delivery_ready"])
            self.assertIn(result["export_status"], ["failed_validation", "failed_texture_application"])

    def test_validator_gates_on_export_status(self):
        """
        Verifies that AssetValidator fails if export_status is not success.
        """
        # Scenario: Exporter failed texture application
        report = self.validator.validate("asset_fail", {
            "export_status": "failed_texture_application",
            "delivery_ready": False,
            "delivery_profile": "mobile_high",
            "poly_count": 50000,
            "all_primitives_have_position": True,
            "all_primitives_have_normal": True,
            "all_textured_primitives_have_texcoord_0": True,
            "bbox": {"width": 10, "height": 10, "depth": 10},
            "ground_offset": 0.0
        }, allow_texture_quality_skip=True)
        
        self.assertEqual(report.final_decision, "fail")
        self.assertIn("export_status", report.blocking_checks)

    def test_validator_strict_texture_integrity(self):
        """
        Verifies texture_integrity_status=complete fails if texture_count=0.
        """
        report = self.validator.validate("asset_hollow_texture", {
            "texture_integrity_status": "complete",
            "texture_applied": True,
            "texture_count": 0, # LIAR!
            "has_uv": True,
            "has_material": True,
            "material_semantic_status": "diffuse_textured",
            "delivery_profile": "mobile_high",
            "poly_count": 50000,
            "all_primitives_have_position": True,
            "all_primitives_have_normal": True,
            "all_textured_primitives_have_texcoord_0": True,
            "bbox": {"width": 10, "height": 10, "depth": 10},
            "ground_offset": 0.0
        }, allow_texture_quality_skip=True)
        
        self.assertEqual(report.final_decision, "fail")
        self.assertIn("texture_application", report.blocking_checks)

    def test_integration_flow_semantic_derivation(self):
        """
        Verifies IntegrationFlow does not mark diffuse_textured unless export proves texture application.
        """
        metadata = NormalizedMetadata(
            bbox_min={"x":0,"y":0,"z":0}, bbox_max={"x":10,"y":10,"z":10},
            pivot_offset={"z":0}, final_polycount=50000
        )
        
        # Scenario: Cleanup has UV, but Exporter did not apply texture (e.g. failed)
        cleanup_stats = {"has_uv": True}
        export_report = {
            "texture_applied": False,
            "texture_count": 0,
            "delivery_ready": False
        }
        
        validator_input = IntegrationFlow.map_metadata_to_validator_input(
            metadata, cleanup_stats=cleanup_stats, export_report=export_report
        )
        
        # Should be "uv_only", NOT "diffuse_textured"
        self.assertEqual(validator_input["material_semantic_status"], "uv_only")

if __name__ == "__main__":
    unittest.main()
