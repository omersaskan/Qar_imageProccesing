import pytest
import os
from pydantic import ValidationError
from modules.shared_contracts.models import ProductPhysicalProfile, ValidationReport, CaptureSession
from modules.shared_contracts.errors import DuplicateAssetError, PathSafetyError, MetadataCorruptionError
from modules.asset_registry.registry import AssetRegistry
from modules.reconstruction_engine.job_manager import JobManager
from modules.shared_contracts.models import ReconstructionJobDraft, AssetMetadata
from modules.utils.path_safety import validate_safe_path

def test_pydantic_bounds_hardening():
    # Negative dimensions should fail
    with pytest.raises(ValidationError):
        ProductPhysicalProfile(real_width_cm=-10, real_depth_cm=10, real_height_cm=10)
    
    # Negative polycounts should fail
    with pytest.raises(ValidationError):
        ValidationReport(
            asset_id="v1", poly_count=-5, texture_status="ok", 
            bbox_reasonable=True, ground_aligned=True, 
            mobile_performance_grade="A", final_decision="pass"
        )
    
    # Invalid coverage score should fail
    with pytest.raises(ValidationError):
        CaptureSession(session_id="S1", product_id="P1", operator_id="O1", coverage_score=1.5)

def test_path_traversal_protection():
    base_dir = "/safe/data"
    # Traversing up and out of base_dir should fail
    with pytest.raises(PathSafetyError):
        validate_safe_path(base_dir, "../../etc/passwd")
    
    # Subdir should be fine
    safe_path = validate_safe_path(base_dir, "jobs/job_1")
    assert "/safe/data/jobs/job_1" in str(safe_path).replace("\\", "/")

def test_duplicate_asset_protection():
    registry = AssetRegistry()
    asset_meta = AssetMetadata(asset_id="v1_same", product_id="p1", version="1.0.0")
    registry.register_asset(asset_meta)
    
    # Registering the same asset_id again should fail
    with pytest.raises(DuplicateAssetError):
        registry.register_asset(asset_meta)

def test_empty_recon_input_protection():
    job_mgr = JobManager(data_root="temp_recon_data")
    draft = ReconstructionJobDraft(
        job_id="job_empty", 
        capture_session_id="s1", 
        product_id="p1", 
        input_frames=[] # Empty list
    )
    # Creating a job with no frames should fail
    with pytest.raises(ValueError, match="No input frames"):
        job_mgr.create_job(draft)
    
    if os.path.exists("temp_recon_data"):
        import shutil
        shutil.rmtree("temp_recon_data")
