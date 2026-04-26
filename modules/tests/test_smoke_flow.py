import pytest
import os
import shutil
from modules.shared_contracts.models import Product, CaptureSession, ReconstructionJobDraft, ProductPhysicalProfile
from modules.shared_contracts.lifecycle import AssetStatus, ReconstructionStatus
from modules.reconstruction_engine.job_manager import JobManager
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.export_pipeline.poster_generator import PosterGenerator
from modules.qa_validation.validator import AssetValidator
from modules.asset_registry.registry import AssetRegistry
from modules.asset_registry.publisher import PackagePublisher
from modules.integration_flow import IntegrationFlow

@pytest.fixture
def smoke_env():
    # Setup temporary directory for smoke test
    test_root = "data_smoke"
    if os.path.exists(test_root):
        shutil.rmtree(test_root)
    os.makedirs(test_root)
    
    yield test_root
    
    if os.path.exists(test_root):
        shutil.rmtree(test_root)

def test_full_factory_smoke_flow(smoke_env):
    """
    Verifies the complete pipeline: Capture -> Recon -> Cleanup -> Validate -> Publish.
    """
    data_root = smoke_env
    # 0. Prep Registry
    registry_path = os.path.join(smoke_env, "registry")
    registry = AssetRegistry(data_root=registry_path)
    publisher = PackagePublisher(registry)
    cleaner = AssetCleaner(data_root=smoke_env)
    job_mgr = JobManager(data_root=smoke_env)
    validator = AssetValidator()
    
    product_id = "smoke_prod_1"
    asset_id = "smoke_asset_v1"
    
    # 1. Capture
    session = CaptureSession(session_id="sess_1", product_id=product_id, operator_id="op_1")
    # Mocking some frames
    session.extracted_frames = ["http://storage/f1.jpg", "http://storage/f2.jpg"]
    session.coverage_score = 0.85
    
    # 2. Reconstruction Draft
    draft = ReconstructionJobDraft(
        job_id="job_1", 
        capture_session_id=session.session_id,
        product_id=product_id,
        input_frames=["f1.jpg", "f2.jpg"]
    )
    job = job_mgr.create_job(draft)
    assert job.status == ReconstructionStatus.QUEUED
    
    # 3. Simulate Reconstruction Output (Raw)
    raw_mesh_path = os.path.join(job.job_dir, "raw_mesh.obj")
    with open(raw_mesh_path, "w") as f:
        # 4 vertices, 4 faces (tetrahedron) to ensure non-zero volume for bbox validation
        f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\nf 2 3 4\nf 3 1 4")
        
    # 4. Cleanup Pipeline
    metadata, stats, cleaned_path = cleaner.process_cleanup(job.job_id, raw_mesh_path)
    assert metadata.final_polycount > 0
    
    # 5. Validation
    report = IntegrationFlow.validate_cleanup_result(asset_id, metadata, validator, allow_texture_quality_skip=True)
    assert report.final_decision in ["pass", "review"]
    
    # 6. Registry & Publish
    # Register the asset first
    from modules.shared_contracts.models import AssetMetadata
    asset_meta = AssetMetadata(asset_id=asset_id, product_id=product_id, version="1.0.0")
    registry.register_asset(asset_meta)
    
    # Publish
    # If it's review, grant approval first
    if report.final_decision == "review":
        registry.grant_approval(asset_id, "review")
        
    physical_profile = ProductPhysicalProfile(
        real_width_cm=10.0, real_depth_cm=10.0, real_height_cm=10.0
    )
    
    package = publisher.publish_package(
        product_id=product_id,
        asset_id=asset_id,
        validation_report=report,
        export_urls={
            "glb_url": "https://cdn.example.com/asset.glb",
            "usdz_url": "https://cdn.example.com/asset.usdz",
            "poster_url": "https://cdn.example.com/poster.png",
            "thumb_url": "https://cdn.example.com/thumb.png",
        },
        physical_profile=physical_profile
    )
    
    assert package.package_status == "ready_for_ar"
    # Use public API to verify state
    history = registry.get_history(product_id)
    assert history[0]["status"] == "published"
    assert history[0]["is_active"] is True
    
    # 7. Verify History Visibility
    history = registry.get_history(product_id)
    assert len(history) == 1
    assert history[0]["asset_id"] == asset_id
