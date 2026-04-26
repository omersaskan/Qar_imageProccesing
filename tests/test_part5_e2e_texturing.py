
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import os
import shutil

from modules.operations.worker import IngestionWorker
from modules.shared_contracts.lifecycle import AssetStatus
from modules.shared_contracts.models import CaptureSession, AssetMetadata
from modules.reconstruction_engine.output_manifest import OutputManifest

class TestPart5E2ETexturing(unittest.TestCase):
    def setUp(self):
        self.test_root = Path("scratch/test_part5_e2e")
        self.test_root.mkdir(parents=True, exist_ok=True)
        
        self.worker = IngestionWorker(data_root=str(self.test_root))
        
        # Create a session in CLEANED state (ready for texturing in _handle_cleanup)
        # Wait, the worker handles cleanup and texturing together in _handle_cleanup.
        # So I should start with RECONSTRUCTED status.
        
        self.session_id = "part5_test_session"
        self.session = CaptureSession(
            session_id=self.session_id,
            status=AssetStatus.RECONSTRUCTED,
            product_id="test_product",
            operator_id="test_op",
            reconstruction_job_id="job_part5",
            reconstruction_manifest_path=str(self.test_root / "reconstructions/job_part5/manifest.json")
        )
        
        # Create dummy reconstruction manifest
        recon_dir = self.test_root / "reconstructions/job_part5"
        recon_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_mesh = recon_dir / "poisson.ply"
        self.raw_mesh.touch()
        
        manifest_data = OutputManifest(
            job_id="job_part5",
            engine_type="colmap",
            mesh_path=str(self.raw_mesh),
            log_path=str(recon_dir / "reconstruction.log"),
            processing_time_seconds=10.0
        )
        with open(self.session.reconstruction_manifest_path, "w") as f:
            json.dump(manifest_data.model_dump(mode="json"), f)
            
        # Register the session
        sessions_dir = self.test_root / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        with open(sessions_dir / f"{self.session_id}.json", "w") as f:
            json.dump(self.session.model_dump(mode="json"), f)

    def tearDown(self):
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    @patch("modules.asset_cleanup_pipeline.cleaner.AssetCleaner.process_cleanup")
    @patch("modules.operations.texturing_service.TexturingService.run")
    @patch("modules.export_pipeline.glb_exporter.GLBExporter.export")
    @patch("modules.export_pipeline.glb_exporter.GLBExporter.inspect_exported_asset")
    @patch("modules.qa_validation.validator.AssetValidator.validate")
    def test_part5_full_textured_flow(self, mock_validate, mock_inspect, mock_export, mock_texturing, mock_cleanup):
        from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
        
        # 1. Mock Cleanup
        cleaned_mesh = self.test_root / "cleaned/job_part5/cleaned_mesh.obj"
        cleaned_mesh.parent.mkdir(parents=True, exist_ok=True)
        cleaned_mesh.touch()
        
        mock_cleanup.return_value = (
            NormalizedMetadata(
                pivot_offset={"x": 0, "y": 0, "z": 0},
                bbox_min={"x": -1, "y": -1, "z": 0},
                bbox_max={"x": 1, "y": 1, "z": 2},
                final_polycount=100
            ),
            {
                "cleaned_mesh_path": str(cleaned_mesh),
                "pre_aligned_mesh_path": str(self.test_root / "cleaned/job_part5/pre_aligned.obj"),
                "cleanup_mode": "standard"
            }, # cleanup_stats
            str(cleaned_mesh)
        )
        
        # 2. Mock Texturing
        textured_mesh = self.test_root / "cleaned/job_part5/textured_aligned_mesh.obj"
        textured_mesh.touch()
        texture_atlas = self.test_root / "cleaned/job_part5/atlas.png"
        texture_atlas.touch()
        
        updated_manifest = OutputManifest(
            job_id="job_part5",
            engine_type="colmap",
            mesh_path=str(textured_mesh),
            textured_mesh_path=str(textured_mesh),
            texture_atlas_paths=[str(texture_atlas)],
            log_path="dummy.log",
            processing_time_seconds=10.0
        )
        updated_manifest.mesh_metadata.has_texture = True
        updated_manifest.mesh_metadata.uv_present = True
        
        from modules.operations.texturing_service import TexturingResult
        mock_texturing.return_value = TexturingResult(
            texturing_status="real",
            cleaned_mesh_path=str(textured_mesh),
            texture_atlas_paths=[str(texture_atlas)],
            manifest=updated_manifest
        )
        
        # 3. Mock Export Metrics
        metrics = {
            "texture_count": 1,
            "material_count": 1,
            "has_uv": True,
            "texture_applied": True
        }
        mock_export.return_value = metrics
        mock_inspect.return_value = metrics
        
        # 4. Mock Validation
        from modules.shared_contracts.models import ValidationReport
        mock_validate.return_value = ValidationReport(
            session_id=self.session_id,
            asset_id="test_asset",
            final_decision="pass",
            delivery_ready=True,
            texture_status="complete",
            material_semantic_status="diffuse_textured",
            poly_count=1000,
            bbox_reasonable=True,
            ground_aligned=True,
            mobile_performance_grade="A",
            checks=[]
        )

        # Run Worker cycle for CLEANED
        self.worker._process_pending_sessions()
        
        updated_session = self.worker.session_manager.get_session(self.session_id)
        self.assertEqual(updated_session.status, AssetStatus.CLEANED)
        
        # Run Worker cycle for EXPORTED
        self.worker._process_pending_sessions()
        updated_session = self.worker.session_manager.get_session(self.session_id)
        self.assertEqual(updated_session.status, AssetStatus.EXPORTED)
        
        # Manually create export_metrics.json (simulating GLBExporter.export behavior)
        metrics_dir = self.test_root / "captures" / self.session_id / "reports"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / "export_metrics.json", "w") as f:
            json.dump(metrics, f)
        
        # Run Worker cycle for VALIDATED
        self.worker._process_pending_sessions()
        updated_session = self.worker.session_manager.get_session(self.session_id)
        if updated_session.status == AssetStatus.FAILED:
             print(f"DEBUG: Session failed at validation. Reason: {updated_session.failure_reason}")

        self.assertEqual(updated_session.status, AssetStatus.VALIDATED)
        self.assertEqual(updated_session.publish_state, "pending")

if __name__ == "__main__":
    unittest.main()
