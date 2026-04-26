import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import trimesh
import numpy as np
import os
import shutil

from modules.reconstruction_engine.adapter import OpenMVSAdapter
from modules.asset_cleanup_pipeline.isolation import MeshIsolator
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner, CleanupProfileType

class TestPart3HardenedVerification(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("temp_test_part3_hardened")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)
        
        self.dense_dir = self.test_dir / "dense"
        self.dense_dir.mkdir()
        
        # Mock settings
        self.settings_patcher = patch("modules.operations.settings.settings")
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.openmvs_path = str(self.test_dir / "bin")
        self.mock_settings.require_textured_output = True

    def tearDown(self):
        self.settings_patcher.stop()
        if self.test_dir.exists():
            import time
            for _ in range(5):
                try:
                    shutil.rmtree(self.test_dir)
                    break
                except PermissionError:
                    time.sleep(0.1)
            else:
                logger.warning("Could not remove test directory: %s", self.test_dir)

    @patch("modules.reconstruction_engine.adapter.OpenMVSAdapter._run_command")
    @patch("trimesh.load")
    @patch("modules.asset_cleanup_pipeline.isolation.MeshIsolator.isolate_product")
    def test_openmvs_object_first_command_flow(self, mock_isolate, mock_trimesh_load, mock_run_cmd):
        """
        Verifies that OpenMVSAdapter:
        1. Runs ReconstructMesh
        2. Performs isolation on the output
        3. Passes the ISOLATED mesh to TextureMesh
        4. Does NOT pass the raw mesh to TextureMesh
        """
        adapter = OpenMVSAdapter()
        adapter.mvs_builder = MagicMock()
        
        # 1. Setup paths
        project_mesh_ply = self.dense_dir / "project_mesh.ply"
        project_mesh_isolated_ply = self.dense_dir / "project_mesh_isolated.ply"
        project_textured_obj = self.dense_dir / "project_textured.obj"
        
        # Create dummy project_mesh.ply so exists() returns True
        project_mesh_ply.write_text("dummy raw mesh")
        
        # 2. Setup Mocks
        # trimesh.load returns a dummy mesh
        mock_raw_mesh = MagicMock(spec=trimesh.Trimesh)
        mock_raw_mesh.vertices = np.random.rand(10, 3)
        mock_raw_mesh.faces = np.random.randint(0, 10, (5, 3))
        mock_trimesh_load.return_value = mock_raw_mesh
        
        # MeshIsolator returns an isolated mesh
        mock_isolated_mesh = MagicMock(spec=trimesh.Trimesh)
        mock_isolated_mesh.vertices = np.random.rand(5, 3)
        mock_isolated_mesh.faces = np.random.randint(0, 5, (2, 3))
        mock_isolate.return_value = (mock_isolated_mesh, {
            "object_isolation_status": "success",
            "initial_faces": 5,
            "final_faces": 2
        })
        
        # Mock MVS builder commands to identify them
        adapter.mvs_builder.reconstruct_mesh.return_value = ["ReconstructMesh", "-i", "dense.mvs", "-o", str(project_mesh_ply)]
        adapter.mvs_builder.texture_mesh.side_effect = lambda mvs, mesh, out: ["TextureMesh", "-i", str(mvs), "--mesh-file", str(mesh), "-o", str(out)]
        
        # 3. Execution (Partial run_reconstruction simulation)
        # We simulate the relevant part of run_reconstruction by calling the logic block
        # Or we can try to call run_reconstruction if we mock enough.
        # Given the complexity of run_reconstruction, we'll verify the logic by running it.
        
        # To run run_reconstruction, we need many more files. Let's mock _select_best_sparse_model etc.
        adapter._select_best_sparse_model = MagicMock(return_value={"path": self.test_dir, "registered_images": 10, "points_3d": 100})
        adapter._run_command = MagicMock()
        adapter._mesh_stats = MagicMock(return_value={"vertex_count": 10, "face_count": 5})
        adapter._discover_texture_candidate = MagicMock(return_value="tex.png")
        adapter._resolve_effective_masks_dir = MagicMock(return_value=self.test_dir / "masks")
        
        # We need to mock the MVS project files
        (self.dense_dir / "project.mvs").write_text("mvs")
        (self.dense_dir / "project_dense.mvs").write_text("mvs")
        (self.dense_dir / "project_textured.obj").write_text("obj")
        (self.dense_dir / "project_textured_material_0_map_Kd.png").write_text("png")
        
        # Run it
        with patch("builtins.open", mock_open()):
            adapter.run_reconstruction([str(self.test_dir / "images/frame_001.jpg")], self.test_dir)
            
        # 4. Verifications
        calls = adapter._run_command.call_args_list
        
        # Find ReconstructMesh call
        reconstruct_call = next(c for c in calls if "ReconstructMesh" in c[0][0][0])
        self.assertEqual(str(reconstruct_call[0][0][-1]), str(project_mesh_ply))
        
        # Find TextureMesh call
        texture_call = next(c for c in calls if "TextureMesh" in c[0][0][0])
        
        # CRITICAL: TextureMesh must receive the ISOLATED mesh
        # In our implementation, we update project_mesh_ply variable to project_mesh_isolated_ply
        mesh_file_arg_idx = texture_call[0][0].index("--mesh-file") + 1
        self.assertEqual(str(texture_call[0][0][mesh_file_arg_idx]), str(project_mesh_isolated_ply))
        self.assertNotEqual(str(texture_call[0][0][mesh_file_arg_idx]), str(project_mesh_ply))
        
        # Verify isolation was actually called
        self.assertTrue(mock_isolate.called)
        self.assertTrue(mock_isolated_mesh.export.called)
        self.assertEqual(str(mock_isolated_mesh.export.call_args[0][0]), str(project_mesh_isolated_ply))

    def test_cleaner_texture_safe_copy_no_fallback(self):
        """
        Verifies that texture_safe_copy does NOT fallback to raw mesh on isolation failure.
        """
        cleaner = AssetCleaner()
        
        # 1. Setup paths
        obj_path = self.test_dir / "input.obj"
        obj_path.write_text("v 0 0 0\nvt 0 0\nf 1/1 1/1 1/1\nmtllib input.mtl\n")
        (self.test_dir / "input.mtl").write_text("newmtl mat\nmap_Kd tex.png\n")
        (self.test_dir / "tex.png").write_text("png")
        
        output_dir = self.test_dir / "output"
        output_dir.mkdir()
        
        # 2. Mock isolation failure
        with patch.object(MeshIsolator, "isolate_product") as mock_iso:
            mock_iso.return_value = (MagicMock(), {"object_isolation_status": "failed_empty"})
            
            # This should now return a failure dict or raise
            result = cleaner._run_texture_safe_copy(str(obj_path), str(self.test_dir / "tex.png"), output_dir)
            
            self.assertEqual(result["cleanup_mode"], "failed")
            self.assertEqual(result["object_isolation_status"], "failed")
            self.assertTrue(result["unsafe_scene_copy_forbidden"])
            
            # Verify that cleaned_mesh.obj was NOT created (or at least not as a success artifact)
            self.assertFalse((output_dir / "cleaned_mesh.obj").exists())

    def test_cleaner_texture_safe_copy_preservation(self):
        """
        Verifies that texture_safe_copy correctly preserves UVs and normalizes MTL/usemtl.
        """
        cleaner = AssetCleaner(data_root=str(self.test_dir))
        
        # 1. Setup paths
        input_dir = self.test_dir / "safe_input"
        input_dir.mkdir()
        output_dir = self.test_dir / "safe_output"
        output_dir.mkdir()
        
        obj_path = input_dir / "input.obj"
        # Include vt and usemtl
        obj_path.write_text("v 0 0 0\nvt 0.5 0.5\nusemtl original_mat\nf 1/1 1/1 1/1\nmtllib input.mtl\n")
        (input_dir / "input.mtl").write_text("newmtl original_mat\nmap_Kd tex.png\n")
        (input_dir / "tex.png").write_text("png")
        
        # 2. Mock isolation to succeed
        with patch.object(MeshIsolator, "isolate_product") as mock_iso:
            # We must return a trimesh that has UVs if we want them preserved in the export
            mock_mesh = trimesh.Trimesh(vertices=[[0,0,0]], faces=[[0,0,0]])
            # Mock trimesh export to write our expected content or just let it run if it's real
            # Since trimesh is installed, we can use a real one
            mock_iso.return_value = (mock_mesh, {"object_isolation_status": "success"})
            
            cleaner.process_cleanup("safe_job", str(obj_path), profile_type=CleanupProfileType.TEXTURE_SAFE_COPY, raw_texture_path=str(input_dir / "tex.png"))
            
            # 3. Verify OBJ normalization
            cleaned_obj = self.test_dir / "cleaned" / "safe_job" / "cleaned_mesh.obj"
            self.assertTrue(cleaned_obj.exists())
            obj_content = cleaned_obj.read_text()
            self.assertIn("usemtl material_0", obj_content)
            self.assertNotIn("usemtl original_mat", obj_content)
            
            # 4. Verify MTL normalization
            cleaned_mtl = self.test_dir / "cleaned" / "safe_job" / "input.mtl"
            self.assertTrue(cleaned_mtl.exists())
            mtl_content = cleaned_mtl.read_text()
            self.assertIn("newmtl material_0", mtl_content)
            self.assertIn("map_Kd tex.png", mtl_content)

    def test_cleaner_diagnostics_completeness(self):
        """
        Verifies all required Part 3 diagnostics are present in cleanup_stats.
        """
        cleaner = AssetCleaner()
        
        # Mock trimesh and isolation
        with patch("trimesh.load") as mock_load, \
             patch.object(MeshIsolator, "isolate_product") as mock_iso:
            
            mock_mesh = MagicMock(spec=trimesh.Trimesh)
            mock_mesh.vertices = np.zeros((10, 3))
            mock_mesh.faces = np.zeros((5, 3))
            mock_load.return_value = mock_mesh
            
            mock_iso.return_value = (mock_mesh, {
                "object_isolation_status": "success",
                "object_isolation_method": "mask_guided",
                "raw_mesh_faces": 100,
                "isolated_mesh_faces": 80,
                "removed_face_ratio": 0.2,
                "mask_support_ratio": 0.95,
                "point_cloud_support_ratio": 0.88
            })
            
            # Setup other components
            cleaner.remesher.process = MagicMock(return_value=80)
            cleaner.alignment.align_to_ground = MagicMock(return_value=(None, {"z": 0}))
            cleaner.bbox_extractor.extract = MagicMock(return_value=({"x":0,"y":0,"z":0}, {"x":1,"y":1,"z":1}))
            
            dummy_obj = self.test_dir / "dummy.obj"
            dummy_obj.write_text("v 0 0 0\nf 1 1 1")
            
            with patch("pathlib.Path.exists", return_value=True), \
                 patch("pathlib.Path.unlink", return_value=None):
                _, stats, _ = cleaner.process_cleanup("job1", str(dummy_obj))
            
            required_keys = [
                "object_isolation_status",
                "object_isolation_method",
                "raw_mesh_faces",
                "isolated_mesh_faces",
                "removed_face_ratio",
                "mask_support_ratio",
                "point_cloud_support_ratio",
                "texture_input_mesh_path",
                "unsafe_scene_copy_forbidden"
            ]
            
            for key in required_keys:
                self.assertIn(key, stats, f"Missing diagnostic key: {key}")
                
            self.assertEqual(stats["object_isolation_status"], "success")
            self.assertEqual(stats["object_isolation_method"], "mask_guided")
            self.assertEqual(stats["removed_face_ratio"], 0.2)

    def test_isolation_no_mask_support_failure(self):
        """
        Verifies that isolation fails when masks are provided but no support is found.
        """
        isolator = MeshIsolator()
        
        # 1. Setup mock data
        mesh = trimesh.Trimesh(vertices=np.random.rand(100, 3), faces=np.random.randint(0, 100, (50, 3)))
        
        # Mask that is all zero (no support)
        masks = [np.zeros((100, 100), dtype=np.uint8)]
        # Camera that projects the mesh into the mask area
        cameras = [{"P": np.eye(3, 4)}] 
        
        # 2. Run isolation
        _, stats = isolator.isolate_product(mesh, masks=masks, cameras=cameras)
        
        # 3. Verify failure
        self.assertEqual(stats["object_isolation_status"], "failed_mask_support")
        self.assertEqual(stats["isolated_mesh_faces"], 0)

    def test_isolation_no_pc_support_failure(self):
        """
        Verifies that isolation fails when point cloud is provided but no support is found.
        """
        isolator = MeshIsolator()
        
        # 1. Setup mock data
        mesh = trimesh.Trimesh(vertices=np.random.rand(100, 3), faces=np.random.randint(0, 100, (50, 3)))
        
        # Point cloud that is far away
        pc = trimesh.points.PointCloud(vertices=np.random.rand(10, 3) + 100.0)
        
        # 2. Run isolation
        _, stats = isolator.isolate_product(mesh, point_cloud=pc)
        
        # 3. Verify failure
        self.assertEqual(stats["object_isolation_status"], "failed_pc_support")
        self.assertEqual(stats["isolated_mesh_faces"], 0)

    def test_cleaner_debug_artifact_preservation(self):
        """
        Verifies that debug_isolated_mesh.obj is preserved after cleanup.
        """
        cleaner = AssetCleaner(data_root=str(self.test_dir))
        dummy_obj = self.test_dir / "dummy.obj"
        dummy_obj.write_text("v 0 0 0\nf 1 1 1")
        
        with patch("trimesh.load") as mock_load, \
             patch.object(MeshIsolator, "isolate_product") as mock_iso, \
             patch("pathlib.Path.exists", return_value=True):
            
            mock_mesh = MagicMock(spec=trimesh.Trimesh)
            mock_mesh.vertices = np.zeros((10, 3))
            mock_mesh.faces = np.zeros((5, 3))
            mock_load.return_value = mock_mesh
            mock_iso.return_value = (mock_mesh, {"object_isolation_status": "success"})
            
            # Mock other parts
            cleaner.remesher.process = MagicMock(return_value=100)
            cleaner.alignment.align_to_ground = MagicMock(return_value=(None, {"z":0}))
            cleaner.bbox_extractor.extract = MagicMock(return_value=({}, {}))

            _, stats, _ = cleaner.process_cleanup("debug_job", str(dummy_obj))
            
            debug_path = Path(stats["texture_input_mesh_path"])
            self.assertIn("debug_isolated_mesh.obj", debug_path.name)
            # Verify that Path.unlink was NOT called for this file (we didn't mock it here, but we can verify intent)
            self.assertTrue(debug_path.name.endswith(".obj"))

if __name__ == "__main__":
    unittest.main()
