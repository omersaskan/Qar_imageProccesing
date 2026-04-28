import json
from pathlib import Path
from modules.integration_flow import IntegrationFlow
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata

def test_integration_flow_mapping():
    metadata = NormalizedMetadata(
        asset_id="test",
        product_id="test_prod",
        bbox_min={"x": 0, "y": 0, "z": 0},
        bbox_max={"x": 1, "y": 1, "z": 1},
        pivot_offset={"z": 0.5},
        final_polycount=1000
    )
    
    export_report = {
        "structural_export_ready": True,
        "delivery_ready": False, # Old flag
        "texture_count": 1,
        "material_count": 1,
        "all_textured_primitives_have_texcoord_0": True
    }
    
    input_data = IntegrationFlow.map_metadata_to_validator_input(
        metadata=metadata,
        export_report=export_report
    )
    
    print(f"Structural Export Ready: {input_data.get('structural_export_ready')}")
    print(f"Delivery Ready (Validator Proxy): {input_data.get('delivery_ready')}")
    
    assert input_data.get("structural_export_ready") is True
    assert input_data.get("delivery_ready") is True
    print("IntegrationFlow mapping test passed!")

if __name__ == "__main__":
    test_integration_flow_mapping()
