from typing import Dict, Any
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
from modules.qa_validation.validator import AssetValidator
from modules.shared_contracts.models import ValidationReport

class IntegrationFlow:
    """
    Standardizes the data bridge between Cleanup and Validation.
    """
    @staticmethod
    def map_metadata_to_validator_input(metadata: NormalizedMetadata, texture_status: str = "complete", profile_name: str = "mobile_high", **overrides) -> Dict[str, Any]:
        """
        Maps NormalizedMetadata (Cleanup output) to AssetValidator input format.
        """
        width = abs(metadata.bbox_max["x"] - metadata.bbox_min["x"])
        height = abs(metadata.bbox_max["y"] - metadata.bbox_min["y"])
        depth = abs(metadata.bbox_max["z"] - metadata.bbox_min["z"])
        
        input_data = {
            "poly_count": metadata.final_polycount,
            "texture_status": texture_status,
            "texture_integrity_status": texture_status,
            "material_semantic_status": "diffuse_textured",
            "delivery_profile": profile_name,
            "bbox": {
                "width": width,
                "height": height,
                "depth": depth
            },
            "ground_offset": metadata.pivot_offset.get("z", 0.0),
        }
        input_data.update(overrides)
        return input_data

    @staticmethod
    def validate_cleanup_result(asset_id: str, metadata: NormalizedMetadata, validator: AssetValidator, allow_texture_quality_skip: bool = False, **kwargs) -> ValidationReport:
        """
        Directly validates the cleanup metadata using the provided validator.
        """
        validator_input = IntegrationFlow.map_metadata_to_validator_input(metadata, **kwargs)
        return validator.validate(asset_id, validator_input, allow_texture_quality_skip=allow_texture_quality_skip)
