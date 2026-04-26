
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import os
import shutil
import trimesh
import numpy as np

from modules.operations.texturing_service import TexturingService
from modules.reconstruction_engine.output_manifest import OutputManifest

class TestTexturingServicePart5(unittest.TestCase):
    def setUp(self):
        self.service = TexturingService()
        self.test_dir = Path("scratch/test_texturing_part5")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy COLMAP workspace
        self.colmap_dir = self.test_dir / "colmap"
        self.dense_dir = self.colmap_dir / "dense"
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_mesh = self.colmap_dir / "mesh.ply"
        self.raw_mesh.touch()
        
        self.manifest = OutputManifest(
            job_id="test_job",
            engine_type="colmap",
            mesh_path=str(self.raw_mesh),
            textured_mesh_path=None,
            texture_path=None,
            log_path="dummy.log",
            processing_time_seconds=0.0
        )
        
        self.cleanup_dir = self.test_dir / "cleanup"
        self.cleanup_dir.mkdir(parents=True, exist_ok=True)
        self.pre_aligned_mesh = self.cleanup_dir / "pre_aligned.obj"
        
        # Create a simple OBJ for pre_aligned_mesh
        with open(self.pre_aligned_mesh, "w") as f:
            f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            
        self.cleanup_stats = {
            "pre_aligned_mesh_path": str(self.pre_aligned_mesh),
            "cleanup_mode": "standard"
        }
        
        self.pivot_offset = {"x": 0.0, "y": 0.0, "z": 0.0}

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @patch("modules.reconstruction_engine.openmvs_texturer.OpenMVSTexturer.run_texturing")
    def test_successful_texturing_flow(self, mock_run):
        # Setup mock results
        texturing_scratch = self.cleanup_dir / "texturing"
        texturing_scratch.mkdir(exist_ok=True)
        
        mock_obj = texturing_scratch / "textured.obj"
        mock_mtl = texturing_scratch / "textured.mtl"
        mock_png = texturing_scratch / "textured_map_Kd.png"
        
        with open(mock_obj, "w") as f:
            f.write("mtllib textured.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl material_0\nf 1/1 2/2 3/3\n")
        
        with open(mock_mtl, "w") as f:
            f.write("newmtl material_0\nmap_Kd textured_map_Kd.png\n")
            
        with open(mock_png, "wb") as f:
            f.write(b"fake_png_data")
            
        mock_run.return_value = {
            "textured_mesh_path": str(mock_obj),
            "texture_atlas_paths": [str(mock_png)],
            "texturing_engine": "openmvs",
            "log_path": str(texturing_scratch / "texturing.log")
        }

        # Execute
        result = self.service.run(
            manifest=self.manifest,
            cleanup_stats=self.cleanup_stats,
            pivot_offset=self.pivot_offset,
            cleaned_mesh_path=str(self.cleanup_dir / "final.obj")
        )

        # 1. Verify status
        self.assertEqual(result.texturing_status, "real")
        
        # 2. Verify manifest updates
        self.assertTrue(result.manifest.mesh_metadata.has_texture)
        self.assertTrue(result.manifest.mesh_metadata.uv_present)
        self.assertIsNotNone(result.manifest.textured_mesh_path)
        self.assertEqual(len(result.manifest.texture_atlas_paths), 1)
        
        # 3. Verify files were relocated to cleanup_dir (parent of cleaned_mesh_path)
        final_obj = Path(result.cleaned_mesh_path)
        self.assertEqual(final_obj.parent, self.cleanup_dir)
        self.assertTrue(final_obj.exists())
        
        final_mtl = self.cleanup_dir / "textured.mtl"
        self.assertTrue(final_mtl.exists())
        
        final_png = self.cleanup_dir / "textured_map_Kd.png"
        self.assertTrue(final_png.exists())

        # 4. Verify OBJ content (vt, mtllib, usemtl)
        with open(final_obj, "r") as f:
            content = f.read()
            self.assertIn("vt ", content)
            self.assertIn("mtllib ", content)
            self.assertIn("usemtl ", content)
            
        # 5. Verify MTL content (map_Kd points to relative filename)
        with open(final_mtl, "r") as f:
            content = f.read()
            self.assertIn("map_Kd textured_map_Kd.png", content)

    @patch("modules.reconstruction_engine.openmvs_texturer.OpenMVSTexturer.run_texturing")
    def test_failed_texturing_flow(self, mock_run):
        # Setup mock failure
        mock_run.side_effect = Exception("TextureMesh crashed")

        # Execute
        result = self.service.run(
            manifest=self.manifest,
            cleanup_stats=self.cleanup_stats,
            pivot_offset=self.pivot_offset,
            cleaned_mesh_path=str(self.cleanup_dir / "final.obj")
        )

        # Verify degraded status
        self.assertEqual(result.texturing_status, "degraded")
        self.assertFalse(result.manifest.mesh_metadata.has_texture)

if __name__ == "__main__":
    unittest.main()
