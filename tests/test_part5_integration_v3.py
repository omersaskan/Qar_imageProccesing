
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import os
import shutil
import cv2
import numpy as np
import trimesh

from modules.operations.texturing_service import TexturingService
from modules.export_pipeline.glb_exporter import GLBExporter
from modules.qa_validation.validator import AssetValidator
from modules.reconstruction_engine.output_manifest import OutputManifest
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata

class TestPart5IntegrationV3(unittest.TestCase):
    def setUp(self):
        self.test_root = Path("scratch/test_part5_integration")
        self.test_root.mkdir(parents=True, exist_ok=True)
        
        # 1. Create a dummy COLMAP workspace for TexturingService to find images/masks
        self.colmap_dir = self.test_root / "colmap"
        self.dense_dir = self.colmap_dir / "dense"
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        (self.colmap_dir / "images").mkdir(exist_ok=True)
        
        # 2. Create the "isolated" mesh that TexturingService will texture
        self.isolated_mesh_path = self.test_root / "isolated.obj"
        box = trimesh.creation.box(extents=[1, 1, 1])
        box.export(str(self.isolated_mesh_path))
        
        # Create raw.ply inside dense
        self.raw_mesh_path = self.dense_dir / "raw.ply"
        self.raw_mesh_path.touch()

        self.cleanup_stats = {
            "pre_aligned_mesh_path": str(self.isolated_mesh_path),
            "cleanup_mode": "standard",
            "isolation": {
                "object_isolation_status": "success",
                "initial_faces": 1000,
                "final_faces": 1000,
                "largest_component_share": 1.0
            },
            "decimation": {"decimation_status": "success"}
        }
        
        self.manifest = OutputManifest(
            job_id="int_job",
            engine_type="colmap",
            mesh_path=str(self.raw_mesh_path),
            log_path="dummy.log",
            processing_time_seconds=1.0
        )
        
        self.pivot_offset = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.cleaned_mesh_path = str(self.test_root / "cleaned_aligned.obj")

    def tearDown(self):
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    @patch("modules.qa_validation.texture_quality.TextureQualityAnalyzer.analyze_path")
    @patch("modules.reconstruction_engine.openmvs_texturer.OpenMVSTexturer.run_texturing")
    def test_full_textured_pipeline_integration(self, mock_run_texturing, mock_analyze):
        mock_analyze.return_value = {"texture_quality_status": "success", "atlas_coverage_ratio": 1.0}
        # SETUP MOCK: OpenMVSTexturer.run_texturing returns real files
        mock_output_dir = self.test_root / "mvs_output"
        mock_output_dir.mkdir(exist_ok=True)
        
        # Create real textured OBJ bundle with trimesh
        box = trimesh.creation.box(extents=[1, 1, 1])
        uvs = np.random.rand(len(box.vertices), 2)
        # Use a colorful pattern to avoid 'flat color' and 'low detail' failures
        tex_img = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                tex_img[i, j] = [i, j, (i+j)//2]
        tex_path = mock_output_dir / "atlas.png"
        cv2.imwrite(str(tex_path), tex_img)
        
        from PIL import Image
        pil_img = Image.fromarray(cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB))
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=pil_img)
        box.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
        
        textured_obj = mock_output_dir / "textured.obj"
        box.export(str(textured_obj))
        
        # Ensure MTL points to atlas.png
        with open(mock_output_dir / "textured.mtl", "w") as f:
            f.write("newmtl material_0\nmap_Kd atlas.png\n")
            
        mock_run_texturing.return_value = {
            "textured_mesh_path": str(textured_obj),
            "texture_atlas_paths": [str(tex_path)],
            "texturing_engine": "openmvs",
            "log_path": "texturing.log"
        }

        # --- STEP 1: TexturingService (REAL) ---
        service = TexturingService()
        tex_result = service.run(
            manifest=self.manifest,
            cleanup_stats=self.cleanup_stats,
            pivot_offset=self.pivot_offset,
            cleaned_mesh_path=self.cleaned_mesh_path
        )
        
        self.assertEqual(tex_result.texturing_status, "real")
        self.assertTrue(os.path.exists(tex_result.cleaned_mesh_path))
        self.assertTrue(tex_result.manifest.mesh_metadata.has_texture)

        # --- STEP 2: GLBExporter (REAL) ---
        exporter = GLBExporter()
        glb_path = self.test_root / "final.glb"
        metadata = NormalizedMetadata(
            pivot_offset=self.pivot_offset,
            bbox_min={"x": -0.5, "y": -0.5, "z": -0.5},
            bbox_max={"x": 0.5, "y": 0.5, "z": 0.5},
            final_polycount=12
        )
        
        export_metrics = exporter.export(
            mesh_path=tex_result.cleaned_mesh_path,
            output_path=str(glb_path),
            profile_name="mobile_high",
            texture_path=tex_result.texture_atlas_paths[0] if tex_result.texture_atlas_paths else None,
            metadata=metadata
        )
        
        self.assertTrue(export_metrics["delivery_ready"])
        self.assertEqual(export_metrics["texture_count"], 1)
        self.assertTrue(export_metrics["all_textured_primitives_have_texcoord_0"])

        # --- STEP 3: AssetValidator (REAL) ---
        validator = AssetValidator()
        val_input = {
            **export_metrics,
            "texture_integrity_status": "complete",
            "material_semantic_status": "diffuse_textured",
            "poly_count": export_metrics["final_face_count"],
            "has_uv": True,
            "has_material": True,
            "texture_count": export_metrics["texture_count"],
            "texture_applied": export_metrics["texture_applied"],
            "texture_path": tex_result.texture_atlas_paths[0],
            "expected_product_color": "colorful",
            "delivery_profile": "mobile_high",
            "filtering_status": "object_isolated",
            "bbox": {"x": 1.0, "y": 1.0, "z": 1.0},
            "ground_offset": 0.0,
            "primitive_attributes": ["POSITION", "NORMAL", "TEXCOORD_0"],
            "cleanup_stats": self.cleanup_stats
        }
        
        report = validator.validate("int_asset", val_input)
        if report.final_decision == "fail":
            print(f"DEBUG VALIDATOR FAIL: {report.blocking_checks}")
            print(f"DEBUG VALIDATOR WARNINGS: {report.warning_checks}")
        
        # FINAL VERIFICATION
        self.assertIn(report.final_decision, ["pass", "review"])
        self.assertEqual(report.material_semantic_status, "diffuse_textured")
        self.assertTrue(report.texture_status == "complete")

if __name__ == "__main__":
    unittest.main()
