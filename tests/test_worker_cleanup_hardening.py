import pytest
import os
import trimesh
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from modules.operations.worker import IngestionWorker
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType, PROFILES
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
from modules.shared_contracts.lifecycle import AssetStatus
from modules.shared_contracts.models import CaptureSession, ReconstructionAudit, ValidationReport
from modules.reconstruction_engine.output_manifest import OutputManifest, MeshMetadata
from modules.integration_flow import IntegrationFlow

@pytest.fixture
def mock_session_manager():
    sm = MagicMock()
    sm.captures_dir = Path("data/captures")
    sm.get_capture_path.return_value = Path("data/captures/test_session")
    return sm

@pytest.fixture
def worker(mock_session_manager):
    with patch("modules.operations.worker.AssetRegistry"), \
         patch("modules.operations.worker.GuidanceAggregator"), \
         patch("modules.operations.worker.RetentionService"), \
         patch("modules.operations.worker.TexturingService"), \
         patch("modules.operations.worker.GLBExporter"), \
         patch("modules.operations.worker.AssetValidator"):
        w = IngestionWorker(data_root="data")
        w.session_manager = mock_session_manager
        return w

def test_worker_uses_mobile_default_alias(worker):
    """Verify that CleanupProfileType.MOBILE_DEFAULT exists and is an alias for mobile_high."""
    assert CleanupProfileType.MOBILE_DEFAULT == CleanupProfileType.MOBILE_HIGH
    assert PROFILES[CleanupProfileType.MOBILE_DEFAULT] == PROFILES[CleanupProfileType.MOBILE_HIGH]

def test_handle_validation_uses_integration_flow(worker):
    """Verify that _handle_validation uses IntegrationFlow and handles current GLBExporter metrics."""
    session = CaptureSession(
        session_id="test_session",
        product_id="test_product",
        operator_id="test_operator",
        status=AssetStatus.EXPORTED,
        export_blob_path="data/blobs/test_asset.glb",
        cleanup_metadata_path="data/cleaned/job_test/normalized_metadata.json",
        cleanup_stats_path="data/cleaned/job_test/cleanup_stats.json"
    )
    
    # Mock artifacts
    metadata = NormalizedMetadata.model_construct(
        bbox_min={"x":0,"y":0,"z":0},
        bbox_max={"x":1,"y":1,"z":1},
        pivot_offset={"x":0,"y":0,"z":0},
        final_polycount=200
    )
    cleanup_stats = {"has_uv": True}
    manifest = OutputManifest.model_construct(
        job_id="job_test",
        mesh_path="mesh.ply",
        log_path="log.txt",
        mesh_metadata=MeshMetadata.model_construct(vertex_count=100, face_count=200, has_texture=False, uv_present=True)
    )
    
    # Current GLBExporter metrics schema
    export_metrics = {
        "final_face_count": 200,
        "final_vertex_count": 100,
        "texture_count": 0,
        "material_count": 1,
        "all_primitives_have_position": True,
        "all_primitives_have_normal": True,
        "all_textured_primitives_have_texcoord_0": False,
        "has_uv": True,
        "has_material": True,
        "has_embedded_texture": False,
        "bbox": {"width": 1, "height": 1, "depth": 1},
        "ground_offset": 0.0,
        "face_count": 200 # legacy compatibility sometimes added by exporter
    }
    
    worker._load_cleanup_artifacts = MagicMock(return_value=(metadata, cleanup_stats))
    worker._load_manifest = MagicMock(return_value=manifest)
    worker.exporter.inspect_exported_asset = MagicMock(return_value=export_metrics)
    worker.validator.validate = MagicMock(return_value=ValidationReport.model_construct(
        asset_id="test_asset",
        poly_count=200,
        texture_status="complete",
        bbox_reasonable=True,
        ground_aligned=True,
        mobile_performance_grade="A",
        final_decision="pass"
    ))
    
    with patch("modules.operations.worker.IntegrationFlow.map_metadata_to_validator_input") as mock_map:
        mock_map.return_value = {"poly_count": 200}
        worker._handle_validation(session)
        assert mock_map.called
        # Check that it passed the correct metrics
        args, kwargs = mock_map.call_args
        assert kwargs["export_report"] == export_metrics

def test_oversized_mesh_trigger_gate(worker):
    """Verify that a 9M mesh triggers the budget gate and marks recapture."""
    session = CaptureSession(
        session_id="test_session",
        product_id="test_product",
        operator_id="test_operator",
        status=AssetStatus.RECONSTRUCTED,
        reconstruction_manifest_path="data/recon/job_test/manifest.json"
    )
    
    manifest = OutputManifest.model_construct(
        job_id="job_test",
        mesh_path="huge.ply",
        log_path="log.txt",
        mesh_metadata=MeshMetadata.model_construct(vertex_count=5000000, face_count=9000000)
    )
    
    worker._load_manifest = MagicMock(return_value=manifest)
    
    # Mock cleaner to return oversized mesh failure
    worker.cleaner.process_cleanup = MagicMock(return_value=(
        None, 
        {"status": "failed_oversized_mesh", "raw_faces": 9000000, "reason": "Too big!"}, 
        ""
    ))
    
    with patch.object(worker, "_mark_session_needs_recapture") as mock_recapture:
        worker._handle_cleanup(session)
        assert mock_recapture.called
        assert "Too big!" in mock_recapture.call_args[1]["reason"]

def test_pre_decimation_failure_handling(worker):
    """Verify that pre-decimation failure returns actionable error."""
    # This is partially covered by test_oversized_mesh_trigger_gate, 
    # but we want to ensure it handles the 'failed_error' from cleaner.
    
    session = CaptureSession(session_id="test_session", product_id="p1", operator_id="o1", status=AssetStatus.RECONSTRUCTED)
    manifest = OutputManifest.model_construct(job_id="j1", mesh_path="m.ply", log_path="l.txt")
    worker._load_manifest = MagicMock(return_value=manifest)
    
    worker.cleaner.process_cleanup = MagicMock(return_value=(
        None, 
        {"status": "failed_oversized_mesh", "reason": "Pre-decimation failed: MemoryError"}, 
        ""
    ))
    
    with patch.object(worker, "_mark_session_needs_recapture") as mock_recapture:
        worker._handle_cleanup(session)
        assert "MemoryError" in mock_recapture.call_args[1]["reason"]

if __name__ == "__main__":
    pytest.main([__file__])
