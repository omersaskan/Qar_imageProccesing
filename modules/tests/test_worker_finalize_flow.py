import unittest
import json
import shutil
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root in path
import sys
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Mocking modules that might have side effects on import or init
# But we need the real classes for some type checks if any
from modules.operations.worker import IngestionWorker
from modules.shared_contracts.models import CaptureSession, AssetMetadata
from modules.shared_contracts.lifecycle import AssetStatus
from modules.reconstruction_engine.output_manifest import OutputManifest
from modules.asset_registry.registry import AssetRegistry

class TestWorkerFinalizeFlow(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_worker_finalize")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Override worker dirs to avoid messing with production data
        self.worker = IngestionWorker()
        self.worker.blobs_dir = self.test_dir / "blobs"
        self.worker.blobs_dir.mkdir(parents=True, exist_ok=True)
        
        # Redirect registry for testing
        self.registry_dir = self.test_dir / "registry"
        self.worker.registry = AssetRegistry(data_root=str(self.registry_dir))
        
        self.session_id = "test_session_123"
        self.product_id = "test_product"
        self.session = CaptureSession(
            session_id=self.session_id,
            product_id=self.product_id,
            operator_id="test_op",
            status=AssetStatus.RECONSTRUCTED
        )
        
        # Mock reconstruction paths
        self.job_id = f"job_{self.session_id}"
        self.recon_dir = Path("data/reconstructions") / self.job_id
        self.recon_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.recon_dir / "manifest.json"
        self.raw_mesh_path = self.recon_dir / "raw_mesh.obj"
        self.raw_mesh_path.write_text("v 0 0 0", encoding="utf-8") # Dummy mesh
        
        self.manifest = OutputManifest(
            job_id=self.job_id,
            mesh_path=str(self.raw_mesh_path),
            log_path="dummy.log",
            processing_time_seconds=10.0,
            is_stub=False
        )
        with open(self.manifest_path, "w") as f:
            f.write(self.manifest.model_dump_json())

    def tearDown(self):
        # Clean up
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        if self.recon_dir.exists():
            shutil.rmtree(self.recon_dir)

    @patch("modules.operations.worker.AssetCleaner")
    @patch("modules.operations.worker.AssetValidator")
    @patch("modules.operations.worker.GLBExporter")
    def test_finalize_ingestion_success(self, MockExporterClass, MockValidatorClass, MockCleanerClass):
        import trimesh
        # Setup mocks
        mock_cleaner = MockCleanerClass.return_value
        mock_validator = MockValidatorClass.return_value
        mock_exporter = MockExporterClass.return_value
        
        # Re-assign to worker instance (since they are created in __init__)
        self.worker.cleaner = mock_cleaner
        self.worker.validator = mock_validator
        self.worker.exporter = mock_exporter
        
        # Mocking cleaner return
        from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
        mock_metadata = NormalizedMetadata(
            bbox_min={"x":-0.5, "y":-0.5, "z":0},
            bbox_max={"x":0.5, "y":0.5, "z":1},
            pivot_offset={"x":0, "y":0, "z":0},
            final_polycount=100
        )
        cleaned_mesh_path = str(self.recon_dir / "cleaned_mesh.obj")
        
        # USE REAL GEOMETRY: min_z = 0.0
        # A 1x1x1 box centered at (0,0,0.5) has bottom at z=0
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        mesh.apply_translation([0, 0, 0.5])
        mesh.export(cleaned_mesh_path)
        
        mock_cleaner.process_cleanup.return_value = (mock_metadata, {"isolation": {"component_count": 1}}, cleaned_mesh_path)
        
        # Mocking validator return
        from modules.shared_contracts.models import ValidationReport
        mock_report = ValidationReport(
            asset_id="test_asset",
            poly_count=100,
            texture_status="complete",
            bbox_reasonable=True,
            ground_aligned=True,
            mobile_performance_grade="A",
            final_decision="pass"
        )
        mock_validator.validate.return_value = mock_report
        
        # Mocking exporter
        mock_exporter.export.return_value = {"filesize": 1024}
        
        # RUN
        self.worker._finalize_ingestion(self.session)
        
        # ASSERTIONS
        # 1. Cleaner called with raw mesh
        mock_cleaner.process_cleanup.assert_called_once()
        
        # 2. Orchestration Assertion: Exporter called with CLEANED mesh path
        mock_exporter.export.assert_called_once()
        _, kwargs = mock_exporter.export.call_args
        self.assertEqual(kwargs['mesh_path'], cleaned_mesh_path)
        self.assertNotEqual(kwargs['mesh_path'], str(self.raw_mesh_path))
        
        # 3. SEMANTIC ASSERTION: Validator received measured residual
        mock_validator.validate.assert_called_once()
        args, kwargs = mock_validator.validate.call_args
        # Robust extraction for positional or keyword arguments
        validation_input = kwargs.get("asset_data") if (kwargs and "asset_data" in kwargs) else args[1]
        
        self.assertIn("ground_offset", validation_input)
        self.assertAlmostEqual(validation_input["ground_offset"], 0.0, places=6)
        
        # 4. Registry update: should be published
        history = self.worker.registry.get_history(self.product_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['status'], "published")

    @patch("modules.operations.worker.GLBExporter")
    @patch("modules.operations.worker.AssetCleaner")
    @patch("modules.operations.worker.AssetValidator")
    def test_finalize_ingestion_gate_fails(self, MockValidatorClass, MockCleanerClass, MockExporterClass):
        import trimesh
        # Setup mocks
        mock_cleaner = MockCleanerClass.return_value
        mock_validator = MockValidatorClass.return_value
        mock_exporter = MockExporterClass.return_value
        
        self.worker.cleaner = mock_cleaner
        self.worker.validator = mock_validator
        self.worker.exporter = mock_exporter
        
        from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
        mock_metadata = NormalizedMetadata(
            bbox_min={"x":-0.5, "y":-0.5, "z":10},
            bbox_max={"x":0.5, "y":0.5, "z":11},
            pivot_offset={"x":0, "y":0, "z":0},
            final_polycount=100
        )
        cleaned_mesh_path = str(self.recon_dir / "failed_mesh.obj")
        
        # USE REAL GEOMETRY: min_z = 10.0
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        mesh.apply_translation([0, 0, 10.5])
        mesh.export(cleaned_mesh_path)
        
        mock_cleaner.process_cleanup.return_value = (mock_metadata, {"isolation": {"component_count": 1}}, cleaned_mesh_path)
        
        from modules.shared_contracts.models import ValidationReport
        mock_report = ValidationReport(
            asset_id="test_asset_fail",
            poly_count=100,
            texture_status="complete",
            bbox_reasonable=True,
            ground_aligned=True,
            mobile_performance_grade="A",
            final_decision="fail",
            contamination_report={"plane_contamination": "fail"}
        )
        mock_validator.validate.return_value = mock_report
        
        # Mock _mark_session_failed
        self.worker._mark_session_failed = MagicMock()
        
        # RUN
        self.worker._finalize_ingestion(self.session)
        
        # ASSERTIONS
        # 1. SEMANTIC ASSERTION: Validator received measured residual (10.0)
        mock_validator.validate.assert_called_once()
        args, kwargs = mock_validator.validate.call_args
        validation_input = kwargs.get("asset_data") if (kwargs and "asset_data" in kwargs) else args[1]
        
        self.assertIn("ground_offset", validation_input)
        self.assertAlmostEqual(validation_input["ground_offset"], 10.0, places=6)
        
        # 2. Orchestration Assertion: Exporter should NOT be called on failure
        mock_exporter.export.assert_not_called()
        
        # 3. Mark session failed called
        self.worker._mark_session_failed.assert_called_once()
        
        # 4. Registry should NOT have a published asset
        prod_data = self.worker.registry._load_product_data(self.product_id)
        self.assertEqual(len(prod_data["assets"]), 0)

if __name__ == "__main__":
    unittest.main()
