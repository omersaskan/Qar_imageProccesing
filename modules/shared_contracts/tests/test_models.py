import pytest
from pydantic import ValidationError
from shared_contracts.models import Product, CaptureSession, ProductPhysicalProfile
from shared_contracts.lifecycle import AssetStatus

def test_product_minimal():
    product = Product(id="P1", name="Lahmacun")
    assert product.id == "P1"
    assert product.name == "Lahmacun"
    assert product.description is None

def test_capture_session_default_status():
    session = CaptureSession(
        session_id="S1",
        product_id="P1",
        operator_id="O1"
    )
    assert session.status == AssetStatus.CREATED

def test_physical_profile_valid():
    profile = ProductPhysicalProfile(
        real_width_cm=30.0,
        real_depth_cm=30.0,
        real_height_cm=10.0
    )
    assert profile.ground_offset_cm == 0.0
    assert profile.recommended_scale_multiplier == 1.0

def test_physical_profile_invalid():
    with pytest.raises(ValidationError):
        # Missing required fields
        ProductPhysicalProfile(real_width_cm=30.0)
