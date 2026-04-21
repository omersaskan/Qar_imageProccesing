import unittest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.operations.worker import IngestionWorker
from modules.shared_contracts.models import CaptureSession
from modules.shared_contracts.lifecycle import AssetStatus
from modules.reconstruction_engine.output_manifest import OutputManifest
from modules.asset_registry.registry import AssetRegistry


def write_dummy_png(path: Path):
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff? \x00\x05\xfe\x02\xfe\xdcD\x05\x13"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


class TestWorkerFinalizeFlow(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_worker_finalize")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.worker = IngestionWorker()
        self.worker.blobs_dir = self.test_dir / "blobs"
        self.worker.blobs_dir.mkdir(parents=True, exist_ok=True)

        self.registry_dir = self.test_dir / "registry"
        self.worker.registry = AssetRegistry(data_root=str(self.registry_dir))

        self.session_id = "test_session_123"
        self.product_id = "test_product"
        self.session = CaptureSession(
            session_id=self.session_id,
            product_id=self.product_id,
            operator_id="test_op",
            status=AssetStatus.RECONSTRUCTED,
        )

        self.job_id = f"job_{self.session_id}"
        self.recon_dir = Path("data/reconstructions") / self.job_id
        self.recon_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.recon_dir / "manifest.json"
        self.raw_mesh_path = self.recon_dir / "raw_mesh.obj"
        self.raw_texture_path = self.recon_dir / "raw_texture.png"

        self.raw_mesh_path.write_text(
            "v 0 0 0\n"
            "v 1 0 0\n"
            "v 0 1 0\n"
            "f 1 2 3\n",
            encoding="utf-8",
        )
        write_dummy_png(self.raw_texture_path)

        self.manifest = OutputManifest(
            job_id=self.job_id,
            mesh_path=str(self.raw_mesh_path),
            texture_path=str(self.raw_texture_path),
            log_path="dummy.log",
            processing_time_seconds=10.0,
            is_stub=False,
        )
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write(self.manifest.model_dump_json())

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        if self.recon_dir.exists():
            shutil.rmtree(self.recon_dir)

    @patch("modules.operations.worker.AssetCleaner")
    @patch("modules.operations.worker.AssetValidator")
    @patch("modules.operations.worker.GLBExporter")
    def test_finalize_ingestion_success(self, MockExporterClass, MockValidatorClass, MockCleanerClass):
        import trimesh

        mock_cleaner = MockCleanerClass.return_value
        mock_validator = MockValidatorClass.return_value
        mock_exporter = MockExporterClass.return_value

        self.worker.cleaner = mock_cleaner
        self.worker.validator = mock_validator
        self.worker.exporter = mock_exporter

        from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
        from modules.shared_contracts.models import ValidationReport

        mock_metadata = NormalizedMetadata(
            bbox_min={"x": -0.5, "y": -0.5, "z": 0.0},
            bbox_max={"x": 0.5, "y": 0.5, "z": 1.0},
            pivot_offset={"x": 0.0, "y": 0.0, "z": 0.0},
            final_polycount=100,
        )

        cleaned_mesh_path = str(self.recon_dir / "cleaned_mesh.obj")
        cleaned_texture_path = str(self.recon_dir / "cleaned_texture.png")

        # real geometry: min_z = 0.0
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        mesh.apply_translation([0, 0, 0.5])  # bottom z = 0
        mesh.export(cleaned_mesh_path)
        write_dummy_png(Path(cleaned_texture_path))

        mock_cleaner.process_cleanup.return_value = (
            mock_metadata,
            {
                "isolation": {
                    "component_count": 1,
                    "initial_faces": 12,
                    "final_faces": 12,
                    "removed_plane_face_share": 0.0,
                    "removed_plane_vertex_ratio": 0.0,
                    "compactness_score": 0.6,
                    "selected_component_score": 0.9,
                },
                "uv_preserved": True,
                "material_preserved": True,
                "cleaned_texture_path": cleaned_texture_path,
            },
            cleaned_mesh_path,
        )

        mock_report = ValidationReport(
            asset_id="test_asset",
            poly_count=100,
            texture_status="complete",
            bbox_reasonable=True,
            ground_aligned=True,
            mobile_performance_grade="A",
            component_count=1,
            largest_component_share=1.0,
            contamination_score=0.0,
            contamination_report={},
            final_decision="pass",
        )
        mock_validator.validate.return_value = mock_report

        mock_exporter.export.return_value = {
            "filesize": 1024,
            "has_uv": True,
            "has_material": True,
            "used_texture_path": cleaned_texture_path,
            "texture_applied_successfully": True,
        }
        mock_exporter.inspect_exported_asset.return_value = {
            "vertex_count": 8,
            "face_count": 12,
            "geometry_count": 1,
            "component_count": 1,
            "has_uv": True,
            "has_material": True,
            "has_embedded_texture": True,
            "texture_integrity_status": "complete",
            "material_integrity_status": "complete",
            "material_semantic_status": "diffuse_textured",
            "texture_count": 1,
            "material_count": 1,
            "texture_integrity_status": "complete",
            "bounds_min": {"x": -0.5, "y": -0.5, "z": 0.0},
            "bounds_max": {"x": 0.5, "y": 0.5, "z": 1.0},
            "bbox": {"x": 1.0, "y": 1.0, "z": 1.0},
            "ground_offset": 0.0,
        }

        self.worker._finalize_ingestion(self.session)

        # cleaner called with raw mesh + raw texture
        mock_cleaner.process_cleanup.assert_called_once()
        _, cleaner_kwargs = mock_cleaner.process_cleanup.call_args
        self.assertEqual(cleaner_kwargs["raw_mesh_path"], str(self.raw_mesh_path))
        self.assertEqual(cleaner_kwargs["raw_texture_path"], str(self.raw_texture_path))

        # exporter must use cleaned mesh + cleaned/original texture
        mock_exporter.export.assert_called_once()
        _, export_kwargs = mock_exporter.export.call_args
        self.assertEqual(export_kwargs["mesh_path"], cleaned_mesh_path)
        self.assertNotEqual(export_kwargs["mesh_path"], str(self.raw_mesh_path))
        self.assertEqual(export_kwargs["texture_path"], cleaned_texture_path)

        # validator must receive measured residual, texture integrity flags
        mock_validator.validate.assert_called_once()
        args, kwargs = mock_validator.validate.call_args
        validation_input = kwargs.get("asset_data") if (kwargs and "asset_data" in kwargs) else args[1]

        self.assertIn("ground_offset", validation_input)
        self.assertAlmostEqual(validation_input["ground_offset"], 0.0, places=6)
        self.assertEqual(validation_input["texture_status"], "complete")
        self.assertTrue(validation_input["texture_path_exists"])
        self.assertTrue(validation_input["has_uv"])
        self.assertTrue(validation_input["has_material"])
        self.assertTrue(validation_input["texture_applied_successfully"])
        self.assertEqual(validation_input["delivery_geometry_count"], 1)
        self.assertEqual(validation_input["delivery_component_count"], 1)

        # registry -> published
        history = self.worker.registry.get_history(self.product_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "published")

        stored_asset = self.worker.registry.get_asset(history[0]["asset_id"])
        self.assertEqual(stored_asset.bbox["min"], {"x": -0.5, "y": -0.5, "z": 0.0})
        self.assertEqual(stored_asset.bbox["max"], {"x": 0.5, "y": 0.5, "z": 1.0})
        self.assertEqual(stored_asset.bbox["dimensions"], {"x": 1.0, "y": 1.0, "z": 1.0})

    @patch("modules.operations.worker.AssetCleaner")
    @patch("modules.operations.worker.AssetValidator")
    @patch("modules.operations.worker.GLBExporter")
    def test_review_asset_stays_draft_and_not_active(self, MockExporterClass, MockValidatorClass, MockCleanerClass):
        import trimesh

        mock_cleaner = MockCleanerClass.return_value
        mock_validator = MockValidatorClass.return_value
        mock_exporter = MockExporterClass.return_value

        self.worker.cleaner = mock_cleaner
        self.worker.validator = mock_validator
        self.worker.exporter = mock_exporter

        from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
        from modules.shared_contracts.models import ValidationReport

        mock_metadata = NormalizedMetadata(
            bbox_min={"x": -0.5, "y": -0.5, "z": 0.0},
            bbox_max={"x": 0.5, "y": 0.5, "z": 1.0},
            pivot_offset={"x": 0.0, "y": 0.0, "z": 0.0},
            final_polycount=100,
        )

        cleaned_mesh_path = str(self.recon_dir / "review_mesh.obj")
        cleaned_texture_path = str(self.recon_dir / "review_texture.png")

        mesh = trimesh.creation.box(extents=[1, 1, 1])
        mesh.apply_translation([0, 0, 0.5])
        mesh.export(cleaned_mesh_path)
        write_dummy_png(Path(cleaned_texture_path))

        mock_cleaner.process_cleanup.return_value = (
            mock_metadata,
            {
                "isolation": {
                    "component_count": 1,
                    "initial_faces": 12,
                    "final_faces": 12,
                    "removed_plane_face_share": 0.0,
                    "removed_plane_vertex_ratio": 0.0,
                    "compactness_score": 0.5,
                    "selected_component_score": 0.8,
                },
                "uv_preserved": True,
                "material_preserved": True,
                "cleaned_texture_path": cleaned_texture_path,
            },
            cleaned_mesh_path,
        )

        mock_validator.validate.return_value = ValidationReport(
            asset_id="test_asset_review",
            poly_count=100,
            texture_status="complete",
            bbox_reasonable=True,
            ground_aligned=True,
            mobile_performance_grade="A",
            component_count=1,
            largest_component_share=1.0,
            contamination_score=0.1,
            contamination_report={"texture_application": "review"},
            final_decision="review",
        )

        mock_exporter.export.return_value = {
            "filesize": 1024,
            "has_uv": True,
            "has_material": True,
            "used_texture_path": cleaned_texture_path,
            "texture_applied_successfully": True,
        }
        mock_exporter.inspect_exported_asset.return_value = {
            "vertex_count": 8,
            "face_count": 12,
            "geometry_count": 1,
            "component_count": 1,
            "has_uv": True,
            "has_material": True,
            "has_embedded_texture": False,
            "bounds_min": {"x": -0.5, "y": -0.5, "z": 0.0},
            "bounds_max": {"x": 0.5, "y": 0.5, "z": 1.0},
            "bbox": {"x": 1.0, "y": 1.0, "z": 1.0},
            "ground_offset": 0.0,
        }

        final_session = self.worker._finalize_ingestion(self.session)

        self.assertEqual(final_session.status, AssetStatus.VALIDATED)
        self.assertEqual(final_session.publish_state, "draft")

        history = self.worker.registry.get_history(self.product_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "draft")
        self.assertFalse(history[0]["is_active"])
        self.assertIsNone(self.worker.registry._get_active_id(self.product_id))

    @patch("modules.operations.worker.GLBExporter")
    @patch("modules.operations.worker.AssetCleaner")
    @patch("modules.operations.worker.AssetValidator")
    def test_finalize_ingestion_gate_fails(self, MockValidatorClass, MockCleanerClass, MockExporterClass):
        import trimesh

        mock_cleaner = MockCleanerClass.return_value
        mock_validator = MockValidatorClass.return_value
        mock_exporter = MockExporterClass.return_value

        self.worker.cleaner = mock_cleaner
        self.worker.validator = mock_validator
        self.worker.exporter = mock_exporter

        from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
        from modules.shared_contracts.models import ValidationReport

        mock_metadata = NormalizedMetadata(
            bbox_min={"x": -0.5, "y": -0.5, "z": 10.0},
            bbox_max={"x": 0.5, "y": 0.5, "z": 11.0},
            pivot_offset={"x": 0.0, "y": 0.0, "z": 0.0},
            final_polycount=100,
        )

        cleaned_mesh_path = str(self.recon_dir / "failed_mesh.obj")
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        mesh.apply_translation([0, 0, 10.5])  # bottom z = 10
        mesh.export(cleaned_mesh_path)

        mock_cleaner.process_cleanup.return_value = (
            mock_metadata,
            {
                "isolation": {
                    "component_count": 6,
                    "initial_faces": 12,
                    "final_faces": 3,
                    "removed_plane_face_share": 0.4,
                    "removed_plane_vertex_ratio": 0.4,
                    "compactness_score": 0.01,
                    "selected_component_score": 0.1,
                },
                "uv_preserved": False,
                "material_preserved": False,
                "cleaned_texture_path": None,
            },
            cleaned_mesh_path,
        )

        mock_report = ValidationReport(
            asset_id="test_asset_fail",
            poly_count=100,
            texture_status="missing",
            bbox_reasonable=True,
            ground_aligned=False,
            mobile_performance_grade="A",
            component_count=6,
            largest_component_share=0.25,
            contamination_score=0.9,
            contamination_report={"plane_contamination": "fail"},
            final_decision="fail",
        )
        mock_validator.validate.return_value = mock_report
        mock_exporter.inspect_exported_asset.return_value = {
            "vertex_count": 8,
            "face_count": 12,
            "geometry_count": 2,
            "component_count": 6,
            "has_uv": False,
            "has_material": False,
            "has_embedded_texture": False,
            "bounds_min": {"x": -0.5, "y": -0.5, "z": 10.0},
            "bounds_max": {"x": 0.5, "y": 0.5, "z": 11.0},
            "bbox": {"x": 1.0, "y": 1.0, "z": 1.0},
            "ground_offset": 10.0,
        }

        self.worker._mark_session_failed = MagicMock()

        self.worker._finalize_ingestion(self.session)

        # semantic assertion: measured residual passed to validator
        mock_validator.validate.assert_called_once()
        args, kwargs = mock_validator.validate.call_args
        validation_input = kwargs.get("asset_data") if (kwargs and "asset_data" in kwargs) else args[1]

        self.assertIn("ground_offset", validation_input)
        self.assertAlmostEqual(validation_input["ground_offset"], 10.0, places=6)
        self.assertEqual(validation_input["delivery_component_count"], 6)

        # export happens before validation in the new lifecycle
        mock_exporter.export.assert_called_once()

        # failed session path
        self.worker._mark_session_failed.assert_called_once()

        # no registry asset
        prod_data = self.worker.registry._load_product_data(self.product_id)
        self.assertEqual(len(prod_data["assets"]), 0)


if __name__ == "__main__":
    unittest.main()
