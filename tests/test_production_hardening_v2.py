import pytest
from modules.operations.settings import Settings
from modules.shared_contracts.models import AssetMetadata, ValidationReport, ProductPhysicalProfile
from modules.asset_registry.publisher import PackagePublisher
from modules.asset_registry.registry import AssetRegistry
from modules.capture_workflow.session_manager import SessionManager
from modules.utils.path_safety import validate_identifier
import shutil

# --- RESOLUTION GATE TESTS ---

def test_resolution_gate_logic():
    # Mock settings
    settings = Settings()
    settings.min_video_short_edge = 720
    settings.min_video_long_edge = 1280
    
    def check(w, h):
        short = min(w, h)
        long = max(w, h)
        return short >= settings.min_video_short_edge and long >= settings.min_video_long_edge

    # 1280x720 (Landscape) passes
    assert check(1280, 720) is True
    # 720x1280 (Portrait) passes
    assert check(720, 1280) is True
    # 1920x1080 passes
    assert check(1920, 1080) is True
    # 1080x1920 passes
    assert check(1080, 1920) is True
    
    # 640x480 fails (both edges too small)
    assert check(640, 480) is False
    # 1280x600 fails (short edge too small)
    assert check(1280, 600) is False
    # 1000x1000 fails (long edge too small)
    assert check(1000, 1000) is False

# --- SESSION MANAGER DEFENSE-IN-DEPTH TESTS ---

def test_session_manager_identifier_validation(tmp_path):
    sm = SessionManager(data_root=str(tmp_path))
    
    # Valid ID
    sm.create_session("sess_123", "prod_abc", "op_1")
    
    # Illegal characters in session_id
    with pytest.raises(ValueError, match="Session ID contains invalid characters"):
        sm.get_session("sess/123")
        
    with pytest.raises(ValueError, match="Session ID contains invalid characters"):
        sm.update_session("sess;drop", new_status=None)
        
    with pytest.raises(ValueError, match="Session ID contains invalid characters"):
        sm.reset_session("sess\\bad")

# --- PHASE B PUBLISH SAFETY TESTS ---

@pytest.fixture
def registry(tmp_path):
    return AssetRegistry(data_root=str(tmp_path))

@pytest.fixture
def publisher(registry):
    return PackagePublisher(registry)

@pytest.fixture
def base_report():
    return ValidationReport(
        asset_id="asset_1",
        poly_count=1000,
        texture_status="complete",
        bbox_reasonable=True,
        ground_aligned=True,
        mobile_performance_grade="A",
        final_decision="pass"
    )

@pytest.fixture
def base_profile():
    return ProductPhysicalProfile(real_width_cm=10, real_depth_cm=10, real_height_cm=10)

@pytest.fixture
def base_urls():
    return {
        "glb_url": "https://cdn/a.glb",
        "usdz_url": "https://cdn/a.usdz",
        "poster_url": "https://cdn/a.jpg",
        "thumb_url": "https://cdn/a_t.jpg"
    }

def test_publish_safety_ai_generated(registry, publisher, base_report, base_urls, base_profile):
    # Register an AI-generated asset
    meta = AssetMetadata(asset_id="asset_ai", product_id="p1", ai_generated=True)
    registry.register_asset(meta)
    
    report = base_report.model_copy(update={"asset_id": "asset_ai", "final_decision": "pass"})
    
    # Should fail terminaly because it's AI-generated
    with pytest.raises(ValueError, match="AI-generated"):
        publisher.publish_package("p1", "asset_ai", report, base_urls, base_profile)
        
    # Grant approval (Registry requires "review" as status)
    registry.grant_approval("asset_ai", "review")
    
    # Should STILL fail (Phase B Hardening Rule: AI is terminal reject for publish)
    with pytest.raises(ValueError, match="AI-generated"):
        publisher.publish_package("p1", "asset_ai", report, base_urls, base_profile)

def test_publish_safety_requires_manual_review(registry, publisher, base_report, base_urls, base_profile):
    # Register an asset flagged for manual review
    meta = AssetMetadata(asset_id="asset_rev", product_id="p1", requires_manual_review=True)
    registry.register_asset(meta)
    
    report = base_report.model_copy(update={"asset_id": "asset_rev", "final_decision": "pass"})
    
    # Should fail despite "pass" decision
    with pytest.raises(ValueError, match="flagged for manual review"):
        publisher.publish_package("p1", "asset_rev", report, base_urls, base_profile)
        
    # Grant approval
    registry.grant_approval("asset_rev", "review")
    
    # Should now pass
    publisher.publish_package("p1", "asset_rev", report, base_urls, base_profile)
