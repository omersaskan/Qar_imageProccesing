from typing import Dict, Any
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
from modules.qa_validation.validator import AssetValidator
from modules.shared_contracts.models import ValidationReport

class IntegrationFlow:
    """
    Standardizes the data bridge between Cleanup and Validation.
    """
    @staticmethod
    def map_metadata_to_validator_input(metadata: NormalizedMetadata, cleanup_stats: Dict[str, Any] = None, export_report: Dict[str, Any] = None, **overrides) -> Dict[str, Any]:
        """
        Maps NormalizedMetadata and pipeline reports to AssetValidator input format.
        """
        width = abs(metadata.bbox_max["x"] - metadata.bbox_min["x"])
        height = abs(metadata.bbox_max["y"] - metadata.bbox_min["y"])
        depth = abs(metadata.bbox_max["z"] - metadata.bbox_min["z"])
        
        input_data = {
            "poly_count": metadata.final_polycount,
            "bbox": {"width": width, "height": height, "depth": depth},
            "ground_offset": metadata.pivot_offset.get("z", 0.0),
            "cleanup_stats": cleanup_stats or {},
            "delivery_profile": (export_report or {}).get("profile", "raw_archive"),
            "material_semantic_status": "geometry_only", # Default
            "texture_integrity_status": "complete",
            "has_uv": (cleanup_stats or {}).get("has_uv", False),
            "has_material": (cleanup_stats or {}).get("has_material", False),
        }
        
        if export_report:
            input_data.update(export_report)
            
            # Derive semantic status honestly from export results
            if export_report.get("texture_applied") and export_report.get("texture_count", 0) > 0:
                input_data["material_semantic_status"] = "diffuse_textured"
            elif (cleanup_stats or {}).get("has_uv"):
                input_data["material_semantic_status"] = "uv_only"
            
            # Ensure accessor flags are present for rules.py
            input_data["all_primitives_have_position"] = export_report.get("all_primitives_have_position", False)
            input_data["all_primitives_have_normal"] = export_report.get("all_primitives_have_normal", False)
            input_data["all_textured_primitives_have_texcoord_0"] = export_report.get("all_textured_primitives_have_texcoord_0", False)
            input_data["delivery_ready"] = export_report.get("delivery_ready", False)

        input_data.update(overrides)
        return input_data

    @staticmethod
    def validate_cleanup_result(asset_id: str, metadata: NormalizedMetadata, validator: AssetValidator, allow_texture_quality_skip: bool = False, **kwargs) -> ValidationReport:
        """
        Directly validates the cleanup metadata using the provided validator.
        """
        validator_input = IntegrationFlow.map_metadata_to_validator_input(metadata, **kwargs)
        return validator.validate(asset_id, validator_input, allow_texture_quality_skip=allow_texture_quality_skip)
