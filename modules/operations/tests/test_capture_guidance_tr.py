import pytest
from modules.operations.guidance import GuidanceAggregator
from modules.shared_contracts.models import AssetStatus, GuidanceSeverity

def test_guidance_turkish_localization():
    aggregator = GuidanceAggregator()
    session_id = "cap_test_tr"
    
    # Test base status (CREATED)
    guidance = aggregator.generate_guidance(session_id, AssetStatus.CREATED)
    assert "İşlemin başlaması için ürün videosunu yükleyin." in guidance.next_action
    
    codes = [m["code"] for m in guidance.messages]
    assert "AWAITING_UPLOAD" in codes
    assert "Video yüklenmesi bekleniyor." in guidance.messages[0]["message"]

def test_guidance_reconstruct_informed_coaching():
    aggregator = GuidanceAggregator()
    session_id = "cap_test_recon"
    
    # Mock low registration (10/50 frames registered)
    reconstruction_stats = {
        "registered_images": 10,
        "input_frames": ["frame"] * 50,
        "sparse_points": 500
    }
    
    guidance = aggregator.generate_guidance(
        session_id, 
        AssetStatus.RECAPTURE_REQUIRED, 
        reconstruction_stats=reconstruction_stats
    )
    
    assert guidance.should_recapture is True
    # Verify diagnostic code
    codes = [m["code"] for m in guidance.messages]
    assert "LOW_RECONSTRUCTABLE_OVERLAP" in codes

def test_guidance_geometric_integration():
    aggregator = GuidanceAggregator()
    session_id = "cap_test_geom"
    
    # Mock coverage report with geometric codes from CoverageAnalyzer
    coverage_report = {
        "overall_status": "insufficient",
        "reasons": ["ORBIT_GAP_LEFT", "WEAK_ORBIT_CONTINUITY"]
    }
    
    guidance = aggregator.generate_guidance(
        session_id, 
        AssetStatus.RECAPTURE_REQUIRED,
        coverage_report=coverage_report
    )
    
    messages = [m["message"] for m in guidance.messages]
    # Check TR mapping of geometric codes
    assert any("sol tarafı yeterince kapsanmamış" in m for m in messages)
    assert any("tur çek" in m for m in messages)
    assert any("Açı geçişleri çok kopuk" in m for m in messages)
    assert any("yumuşak ve sürekli bir yörüngeyle" in m for m in messages)

def test_guidance_next_action_turkish():
    aggregator = GuidanceAggregator()
    
    # Ready for review
    guidance = aggregator.generate_guidance("s1", AssetStatus.VALIDATED)
    assert "Dashboard'u açarak 3D modeli inceleyin" in guidance.next_action
    
    # Recapture
    guidance = aggregator.generate_guidance("s2", AssetStatus.RECAPTURE_REQUIRED)
    assert "Yeniden çekim gerekli" in guidance.next_action
