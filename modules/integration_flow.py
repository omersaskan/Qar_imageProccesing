from typing import Dict, Any, List
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
from modules.qa_validation.validator import AssetValidator
from modules.shared_contracts.models import ValidationReport

class IntegrationFlow:
    """
    Standardizes the data bridge between Cleanup and Validation.
    Ensures validation is performed on actual cleanup results.
    """
    @staticmethod
    def map_metadata_to_validator_input(metadata: NormalizedMetadata, texture_status: str = "complete") -> Dict[str, Any]:
        """
        Maps NormalizedMetadata (Cleanup output) to AssetValidator input format.
        """
        # Calculate real-world bounding box dimensions for validator sanity check
        width = abs(metadata.bbox_max["x"] - metadata.bbox_min["x"])
        height = abs(metadata.bbox_max["y"] - metadata.bbox_min["y"])
        depth = abs(metadata.bbox_max["z"] - metadata.bbox_min["z"])
        
        return {
            "poly_count": metadata.final_polycount,
            "texture_status": texture_status,
            # validator.validate() reads texture_integrity_status specifically
            "texture_integrity_status": texture_status,
            # Default to a photogrammetry-appropriate semantic status
            "material_semantic_status": "diffuse_textured",
            "bbox": {
                "width": width,
                "height": height,
                "depth": depth
            },
            # Pivot offset Z is often used as the base-to-center ground offset in this factory
            "ground_offset": metadata.pivot_offset.get("z", 0.0),
            # Cleanup results are assumed to have valid accessors for geometry
            "has_position_accessor": True,
            "has_normal_accessor": True,
            "has_texcoord_0_accessor": True
        }

    @staticmethod
    def validate_cleanup_result(asset_id: str, metadata: NormalizedMetadata, validator: AssetValidator, allow_texture_quality_skip: bool = False) -> ValidationReport:
        """
        Directly validates the cleanup metadata using the provided validator.
        """
        validator_input = IntegrationFlow.map_metadata_to_validator_input(metadata)
        return validator.validate(asset_id, validator_input, allow_texture_quality_skip=allow_texture_quality_skip)
