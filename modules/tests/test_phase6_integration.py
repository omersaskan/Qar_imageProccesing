import pytest
import os
import json
from modules.asset_registry.registry import AssetRegistry
from modules.asset_registry.publisher import PackagePublisher
from modules.qa_validation.validator import AssetValidator
from modules.qa_validation.review_actions import ReviewManager
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
from modules.integration_flow import IntegrationFlow
from modules.shared_contracts.models import AssetMetadata, ValidationReport, ProductPhysicalProfile
from modules.operations.telemetry import OperationalTelemetry, FailureCodes

@pytest.fixture
def setup_system():
    registry = AssetRegistry()
    publisher = PackagePublisher(registry)
    validator = AssetValidator()
    review_manager = ReviewManager(registry)
    telemetry = OperationalTelemetry(log_path="temp_operations_log.json")
    
    yield registry, publisher, validator, review_manager, telemetry
    
    if os.path.exists("temp_operations_log.json"):
        os.remove("temp_operations_log.json")

def test_review_policy_enforcement(setup_system):
    registry, publisher, validator, review_manager, telemetry = setup_system
    
    product_id = "prod_1"
    asset_id = "v1_review"
    metadata = AssetMetadata(asset_id=asset_id, product_id=product_id, version="1.0.0")
    registry.register_asset(metadata)
    
    # Simulate a 'review' validation result (polycount 75k)
    report = ValidationReport(
        asset_id=asset_id, poly_count=75000, texture_status="complete",
        bbox_reasonable=True, ground_aligned=True,
        mobile_performance_grade="C", final_decision="review"
    )
    
    # 1. Attempt publish without approval -> Should fail
    with pytest.raises(ValueError, match="lacks manual approval"):
        publisher.publish_package(product_id, asset_id, report, {}, ProductPhysicalProfile(real_width_cm=1, real_depth_cm=1, real_height_cm=1))
        telemetry.log_failure("PUBLISHER", asset_id, FailureCodes.ERR_PUBLISH_BLOCKED_REVIEW, "No approval")

    # 2. Grant approval
    review_manager.approve(asset_id, report)
    assert registry.has_approval(asset_id) is True
    
    # 3. Attempt publish with approval -> Should succeed
    dummy_urls = {
        "glb_url": "https://cdn.example.com/asset.glb",
        "usdz_url": "https://cdn.example.com/asset.usdz",
        "poster_url": "https://cdn.example.com/poster.jpg",
        "thumb_url": "https://cdn.example.com/thumb.jpg",
    }
    package = publisher.publish_package(product_id, asset_id, report, dummy_urls, ProductPhysicalProfile(real_width_cm=1, real_depth_cm=1, real_height_cm=1))
    assert package.package_status == "ready_for_ar"
    assert registry.publish_states[asset_id] == "published"

def test_fail_status_is_terminal(setup_system):
    registry, publisher, validator, review_manager, telemetry = setup_system
    
    asset_id = "v1_fail"
    report = ValidationReport(
        asset_id=asset_id, poly_count=150000, texture_status="complete",
        bbox_reasonable=True, ground_aligned=True,
        mobile_performance_grade="D", final_decision="fail"
    )
    
    # Try to approve a 'fail' asset -> Should fail
    with pytest.raises(ValueError, match="not 'review'"):
        review_manager.approve(asset_id, report)
        
    # Try to publish a 'fail' asset -> Should fail
    with pytest.raises(ValueError, match="terminal"):
        publisher.publish_package("p1", asset_id, report, {}, ProductPhysicalProfile(real_width_cm=1, real_depth_cm=1, real_height_cm=1))

def test_cleanup_validation_bridge(setup_system):
    registry, _, validator, _, _ = setup_system
    
    # Mock Cleanup output
    cleanup_metadata = NormalizedMetadata(
        bbox_min={"x": -10, "y": -10, "z": 0},
        bbox_max={"x": 10, "y": 10, "z": 20},
        pivot_offset={"x": 0, "y": 0, "z": 0.5},
        final_polycount=45000
    )
    
    # Use the bridge
    report = IntegrationFlow.validate_cleanup_result("asset_123", cleanup_metadata, validator)
    
    assert report.final_decision == "pass"
    assert report.poly_count == 45000
    assert report.mobile_performance_grade == "B"

def test_telemetry_logging(setup_system):
    _, _, _, _, telemetry = setup_system
    
    telemetry.log_failure("TEST", "job_1", FailureCodes.ERR_RECON_RUNTIME, "Out of memory")
    telemetry.log_action("asset_1", "manual_review", {"result": "approved"})
    
    with open("temp_operations_log.json", "r") as f:
        logs = json.load(f)
        
    assert len(logs) == 2
    assert logs[0]["failure_code"] == "ERR_RECON_RUNTIME"
    assert logs[1]["action"] == "manual_review"
