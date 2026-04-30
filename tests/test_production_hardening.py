import pytest
from modules.qa_validation.rules import normalize_status, validate_texture_quality
from modules.asset_registry.registry import AssetRegistry
from modules.asset_registry.publisher import PackagePublisher
from modules.shared_contracts.models import ValidationReport, AssetMetadata, ProductPhysicalProfile
from pathlib import Path
import json
import shutil

# --- TEXTURE VALIDATION REGRESSION TESTS ---

def test_texture_status_normalization():
    # pass cases
    assert normalize_status("pass") == "pass"
    assert normalize_status("success") == "pass"
    assert normalize_status("clean") == "pass"
    
    # review cases
    assert normalize_status("review") == "review"
    assert normalize_status("warning") == "review"
    assert normalize_status("degraded") == "review"
    
    # fail cases
    assert normalize_status("fail") == "fail"
    assert normalize_status("failed") == "fail"
    assert normalize_status("contaminated") == "fail"
    assert normalize_status("invalid") == "fail"
    
    # default case
    assert normalize_status("unknown_status") == "fail"

def test_validate_texture_quality_mapping():
    assert validate_texture_quality({"texture_quality_status": "clean"}) == "pass"
    assert validate_texture_quality({"texture_quality_status": "warning"}) == "review"
    assert validate_texture_quality({"texture_quality_status": "contaminated"}) == "fail"
    assert validate_texture_quality({}) == "fail" # Missing data defaults to fail

# --- REGISTRY PUBLISH HARDENING TESTS ---

@pytest.fixture
def temp_registry(tmp_path):
    reg = AssetRegistry(data_root=str(tmp_path))
    return reg

@pytest.fixture
def publisher(temp_registry):
    return PackagePublisher(temp_registry)

@pytest.fixture
def sample_metadata():
    return AssetMetadata(
        asset_id="asset_123",
        product_id="prod_abc",
        version="v1",
        bbox={"x": 1.0, "y": 1.0, "z": 1.0}
    )

@pytest.fixture
def sample_report():
    return ValidationReport(
        asset_id="asset_123",
        poly_count=1000,
        texture_status="complete",
        bbox_reasonable=True,
        ground_aligned=True,
        mobile_performance_grade="A",
        final_decision="pass"
    )

@pytest.fixture
def sample_profile():
    return ProductPhysicalProfile(
        real_width_cm=10,
        real_depth_cm=10,
        real_height_cm=10
    )

@pytest.fixture
def sample_urls():
    return {
        "glb_url": "https://cdn.example.com/asset.glb",
        "usdz_url": "https://cdn.example.com/asset.usdz",
        "poster_url": "https://cdn.example.com/poster.jpg",
        "thumb_url": "https://cdn.example.com/thumb.jpg"
    }

def test_publish_pass_success(temp_registry, publisher, sample_metadata, sample_report, sample_urls, sample_profile):
    temp_registry.register_asset(sample_metadata)
    
    package = publisher.publish_package(
        "prod_abc", "asset_123", sample_report, sample_urls, sample_profile
    )
    
    assert package.validation_status == "pass"
    assert package.package_status == "ready_for_ar"
    
    # Check registry state
    history = temp_registry.get_history("prod_abc")
    asset_info = next(h for h in history if h["asset_id"] == "asset_123")
    assert asset_info["status"] == "published"
    assert asset_info["is_active"] is True
    
    # Check audit log
    audit_actions = [log["action"] for log in asset_info["audit"]]
    assert "published" in audit_actions

def test_publish_fail_rejection(temp_registry, publisher, sample_metadata, sample_report, sample_urls, sample_profile):
    temp_registry.register_asset(sample_metadata)
    sample_report.final_decision = "fail"
    
    with pytest.raises(ValueError, match="failed validation"):
        publisher.publish_package("prod_abc", "asset_123", sample_report, sample_urls, sample_profile)

def test_publish_review_requires_approval(temp_registry, publisher, sample_metadata, sample_report, sample_urls, sample_profile):
    temp_registry.register_asset(sample_metadata)
    sample_report.final_decision = "review"
    
    # Should fail without approval
    with pytest.raises(ValueError, match="lacks manual approval"):
        publisher.publish_package("prod_abc", "asset_123", sample_report, sample_urls, sample_profile)
    
    # Grant approval
    temp_registry.grant_approval("asset_123", "review")
    
    # Should now succeed
    publisher.publish_package("prod_abc", "asset_123", sample_report, sample_urls, sample_profile)
    assert temp_registry._get_active_id("prod_abc") == "asset_123"

def test_publish_updates_active_pointer_atomically(temp_registry, publisher, sample_metadata, sample_report, sample_urls, sample_profile):
    # Register v1
    v1_meta = sample_metadata.model_copy(update={"asset_id": "asset_v1", "version": "v1"})
    temp_registry.register_asset(v1_meta)
    
    # Publish v1
    report_v1 = sample_report.model_copy(update={"asset_id": "asset_v1"})
    publisher.publish_package("prod_abc", "asset_v1", report_v1, sample_urls, sample_profile)
    assert temp_registry._get_active_id("prod_abc") == "asset_v1"
    
    # Register v2
    v2_meta = sample_metadata.model_copy(update={"asset_id": "asset_v2", "version": "v2"})
    temp_registry.register_asset(v2_meta)
    
    # Publish v2
    report_v2 = sample_report.model_copy(update={"asset_id": "asset_v2"})
    publisher.publish_package("prod_abc", "asset_v2", report_v2, sample_urls, sample_profile)
    
    # Active should now be v2
    assert temp_registry._get_active_id("prod_abc") == "asset_v2"
    
    # History should show both, but only v2 active
    history = temp_registry.get_history("prod_abc")
    assert any(h["asset_id"] == "asset_v1" and h["status"] == "published" and not h["is_active"] for h in history)
    assert any(h["asset_id"] == "asset_v2" and h["status"] == "published" and h["is_active"] for h in history)

def test_publish_requires_real_urls(publisher, sample_report, sample_profile):
    bad_urls = {
        "glb_url": "https://missing",
        "usdz_url": "https://cdn.example.com/asset.usdz",
        "poster_url": "https://cdn.example.com/poster.jpg",
        "thumb_url": "https://cdn.example.com/thumb.jpg"
    }
    with pytest.raises(ValueError, match="placeholder export artifacts are not allowed"):
        publisher.publish_package("prod_abc", "asset_123", sample_report, bad_urls, sample_profile)
