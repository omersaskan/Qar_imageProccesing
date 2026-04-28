from pathlib import Path
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
            "delivery_profile": (cleanup_stats or {}).get("delivery_profile") or (export_report or {}).get("profile", "raw_archive"),
            "material_semantic_status": "geometry_only", # Default
            "texture_integrity_status": "missing", # Default
            "has_uv": False,
            "has_material": False,
        }
        
        # 1. Start with cleanup_stats truth
        if cleanup_stats:
            input_data["has_uv"] = bool(cleanup_stats.get("has_uv", False))
            input_data["has_material"] = bool(cleanup_stats.get("has_material", False))
            input_data["texture_integrity_status"] = cleanup_stats.get("texture_integrity_status", "missing")
            input_data["material_semantic_status"] = cleanup_stats.get("material_semantic_status", "geometry_only")
            
            # SPRINT 5C: Load and merge texture quality metrics
            report_path = cleanup_stats.get("texture_quality_report_path")
            if report_path and Path(report_path).exists():
                import json
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        report_data = json.load(f)
                        input_data.update(report_data)
                        # Ensure the validator sees the status
                        if "status" in report_data and "texture_quality_status" not in input_data:
                            input_data["texture_quality_status"] = report_data["status"]
                except Exception:
                    pass

            # Map texture path for analysis fallback
            if "texture_path" not in input_data:
                input_data["texture_path"] = cleanup_stats.get("cleaned_texture_path") or cleanup_stats.get("texture_path")

            # Map Semantic Isolation Metrics
            isolation = cleanup_stats.get("isolation", {})
            input_data["mask_support_ratio"] = isolation.get("mask_support_ratio", 0.0)
            input_data["point_cloud_support_ratio"] = isolation.get("point_cloud_support_ratio", 0.0)
            input_data["supported_view_count"] = isolation.get("supported_view_count", 0)
            input_data["isolation_confidence"] = isolation.get("isolation_confidence", 0.0)
            input_data["component_count"] = isolation.get("component_count", 1)
            input_data["largest_component_share"] = isolation.get("largest_component_share", 1.0)

        # 2. Override with final export truth (The Ultimate Truth)
        if export_report:
            input_data.update(export_report)
            
            final_tex_count = export_report.get("texture_count", 0)
            final_mat_count = export_report.get("material_count", 0)
            final_uv_accessor = export_report.get("all_textured_primitives_have_texcoord_0", False)
            
            if final_tex_count > 0 and final_mat_count > 0 and final_uv_accessor:
                input_data["has_uv"] = True
                input_data["has_material"] = True
                input_data["texture_integrity_status"] = "complete"
                input_data["material_semantic_status"] = "diffuse_textured"
                input_data["texture_applied"] = True # Force for validator
            
            input_data["all_primitives_have_position"] = export_report.get("all_primitives_have_position", False)
            input_data["all_primitives_have_normal"] = export_report.get("all_primitives_have_normal", False)
            input_data["all_textured_primitives_have_texcoord_0"] = final_uv_accessor
            
            # ROOT CAUSE FIX: Carry structural_export_ready truth
            input_data["structural_export_ready"] = export_report.get("structural_export_ready", export_report.get("delivery_ready", False))
            input_data["delivery_ready"] = input_data["structural_export_ready"] # Proxy for validator rules

        input_data.update(overrides)
        
        # Final cleanup for texture_path
        if not input_data.get("texture_path") and (cleanup_stats or {}).get("cleaned_texture_path"):
            input_data["texture_path"] = cleanup_stats["cleaned_texture_path"]

        return input_data

    @staticmethod
    def validate_cleanup_result(asset_id: str, metadata: NormalizedMetadata, validator: AssetValidator, allow_texture_quality_skip: bool = False, **kwargs) -> ValidationReport:
        """
        Directly validates the cleanup metadata using the provided validator.
        """
        validator_input = IntegrationFlow.map_metadata_to_validator_input(metadata, **kwargs)
        return validator.validate(asset_id, validator_input, allow_texture_quality_skip=allow_texture_quality_skip)
