import pytest
from modules.asset_registry.registry import AssetRegistry
from modules.asset_registry.publisher import PackagePublisher
from modules.shared_contracts.models import AssetMetadata, ValidationReport, ProductPhysicalProfile
from pydantic import HttpUrl

def test_registry_registration():
    registry = AssetRegistry()
    asset_id = "asset_v1"
    product_id = "product_123"
    metadata = AssetMetadata(
        asset_id=asset_id,
        product_id=product_id,
        version="1.0.0",
        bbox={"width": 10.0, "height": 20.0, "depth": 5.0},
        pivot_offset={"x": 5.0, "y": 0.0, "z": 2.5}
    )
    
    registry.register_asset(metadata)
    assert registry.get_asset(asset_id) == metadata
    assert asset_id in registry.product_versions[product_id]

def test_registry_active_version():
    registry = AssetRegistry()
    product_id = "product_123"
    metadata_v1 = AssetMetadata(asset_id="v1", product_id=product_id, version="1.0.0")
    metadata_v2 = AssetMetadata(asset_id="v2", product_id=product_id, version="2.0.0")
    
    registry.register_asset(metadata_v1)
    registry.register_asset(metadata_v2)
    
    registry.set_active_version(product_id, "v1")
    assert registry.active_versions[product_id] == "v1"
    
    # Switch
    registry.set_active_version(product_id, "v2")
    assert registry.active_versions[product_id] == "v2"

def test_registry_rollback():
    registry = AssetRegistry()
    product_id = "product_123"
    metadata_v1 = AssetMetadata(asset_id="v1", product_id=product_id, version="1.0.0")
    metadata_v2 = AssetMetadata(asset_id="v2", product_id=product_id, version="2.0.0")
    
    registry.register_asset(metadata_v1)
    registry.register_asset(metadata_v2)
    registry.set_active_version(product_id, "v2")
    
    rolled_back = registry.rollback_version(product_id)
    assert rolled_back == "v1"
    assert registry.active_versions[product_id] == "v1"

def test_publisher_gate_fail():
    registry = AssetRegistry()
    publisher = PackagePublisher(registry)
    
    asset_id = "asset_fail"
    product_id = "prod_fail"
    metadata = AssetMetadata(asset_id=asset_id, product_id=product_id, version="1.0.0")
    registry.register_asset(metadata)
    
    fail_report = ValidationReport(
        asset_id=asset_id,
        poly_count=150_000,
        texture_status="missing_critical",
        bbox_reasonable=False,
        ground_aligned=False,
        mobile_performance_grade="D",
        final_decision="fail"
    )
    
    with pytest.raises(ValueError, match="failed validation"):
        publisher.publish_package(
            product_id=product_id,
            asset_id=asset_id,
            validation_report=fail_report,
            export_urls={},
            physical_profile=ProductPhysicalProfile(real_width_cm=10, real_depth_cm=10, real_height_cm=10)
        )

def test_publisher_successful_publish():
    registry = AssetRegistry()
    publisher = PackagePublisher(registry)
    
    asset_id = "asset_pass"
    product_id = "prod_pass"
    metadata = AssetMetadata(asset_id=asset_id, product_id=product_id, version="1.0.0")
    registry.register_asset(metadata)
    
    pass_report = ValidationReport(
        asset_id=asset_id,
        poly_count=10_000,
        texture_status="complete",
        bbox_reasonable=True,
        ground_aligned=True,
        mobile_performance_grade="A",
        final_decision="pass"
    )
    
    export_urls = {
        "glb_url": "http://cdn.com/model.glb",
        "usdz_url": "http://cdn.com/model.usdz",
        "poster_url": "http://cdn.com/poster.png",
        "thumb_url": "http://cdn.com/thumb.png"
    }
    
    profile = ProductPhysicalProfile(real_width_cm=10, real_depth_cm=10, real_height_cm=10)
    
    package = publisher.publish_package(
        product_id=product_id,
        asset_id=asset_id,
        validation_report=pass_report,
        export_urls=export_urls,
        physical_profile=profile
    )
    
    assert package.package_status == "ready_for_ar"
    assert registry.publish_states[asset_id] == "published"
    assert registry.active_versions[product_id] == asset_id
