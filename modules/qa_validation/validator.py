from typing import Dict, Any, List

from modules.shared_contracts.models import ValidationReport
from .rules import (
    ValidationThresholds,
    validate_polycount_by_profile,
    validate_texture_integrity,
    validate_bbox,
    validate_ground_alignment,
    validate_contamination,
    validate_delivery_mesh,
    validate_texture_quality,
    validate_material_semantics,
    validate_accessors,
    validate_object_filtering,
    validate_decimation,
    validate_export_delivery_status,
)
from .texture_quality import TextureQualityAnalyzer


class AssetValidator:
    def __init__(self, thresholds: ValidationThresholds = None):
        self.thresholds = thresholds or ValidationThresholds()

    def validate(self, asset_id: str, asset_data: Dict[str, Any], allow_texture_quality_skip: bool = False) -> ValidationReport:
        """
        Validates an asset against delivery rules.
        """
        profile_name = asset_data.get("delivery_profile", "raw_archive")
        poly_count = asset_data.get("poly_count", 0)
        
        poly_decision = validate_polycount_by_profile(poly_count, profile_name)
        semantic_status = asset_data.get("material_semantic_status", "geometry_only")
        
        bbox_decision = validate_bbox(asset_data.get("bbox", {}), self.thresholds)
        ground_decision = validate_ground_alignment(asset_data.get("ground_offset", 99.0), self.thresholds)

        cleanup_stats = asset_data.get("cleanup_stats", {})
        iso_stats = cleanup_stats.get("isolation", {})
        decimation_stats = cleanup_stats.get("decimation", {})

        contamination_decisions = validate_contamination(cleanup_stats, self.thresholds)
        texture_integrity_decisions = validate_texture_integrity(asset_data, self.thresholds)
        delivery_decisions = validate_delivery_mesh(asset_data, self.thresholds)
        texturing_status = asset_data.get("texture_integrity_status", "missing")
        if texturing_status == "complete" or asset_data.get("texturing_status") == "real":
            texturing_truth = "real"
        else:
            texturing_truth = "absent"

        decimation_decision = validate_decimation(decimation_stats, texturing_status=texturing_truth)

        # 1. Texture Quality Analysis
        texture_quality_stats = {}
        texture_path = asset_data.get("texture_path")
        expected_color = asset_data.get("expected_product_color", "unknown")
        
        if texture_path:
            analyzer = TextureQualityAnalyzer(thresholds=self.thresholds)
            texture_quality_stats = analyzer.analyze_path(texture_path, expected_product_color=expected_color)
            quality_decision = validate_texture_quality(texture_quality_stats)
        else:
            if "texture_quality_status" in asset_data:
                 quality_decision = validate_texture_quality(asset_data)
                 texture_quality_stats = asset_data
            else:
                 if allow_texture_quality_skip:
                     quality_decision = "pass"
                 else:
                     quality_decision = "pass" if semantic_status == "geometry_only" else "fail"
                     texture_quality_stats = {"texture_quality_status": "fail", "texture_quality_reasons": ["MISSING_TEXTURE_QUALITY_METRICS"]}

        # 2. Decision Aggregation
        checks = {
            "polycount": poly_decision,
            "bbox": bbox_decision,
            "ground_alignment": ground_decision,
            "texture_quality": quality_decision,
            "material_semantics": validate_material_semantics(semantic_status),
            "decimation": decimation_decision,
            "object_filtering": validate_object_filtering(asset_data)
        }
        
        for k, v in contamination_decisions.items(): checks[f"contamination_{k}"] = v
        for k, v in texture_integrity_decisions.items(): checks[f"texture_{k}"] = v
        for k, v in delivery_decisions.items(): checks[f"delivery_{k}"] = v

        accessor_decisions = validate_accessors(asset_data)
        for k, v in accessor_decisions.items(): checks[k] = v

        export_decisions = validate_export_delivery_status(asset_data)
        for k, v in export_decisions.items(): checks[k] = v

        blocking_checks = [k for k, v in checks.items() if v == "fail"]
        warning_checks = [k for k, v in checks.items() if v == "review"]
        passed_checks = [k for k, v in checks.items() if v == "pass"]

        if blocking_checks:
            final_decision = "fail"
        elif warning_checks:
            final_decision = "review"
        else:
            final_decision = "pass"

        # 3. Delivery Strategy
        is_mobile_ready = False
        delivery_status = "pending"
        
        if profile_name == "raw_archive":
            delivery_status = "archive_only"
            is_mobile_ready = False
        elif final_decision == "pass":
            delivery_status = "delivery_ready"
            is_mobile_ready = True
        elif final_decision == "review":
            delivery_status = "review"
            is_mobile_ready = True # Review assets are mobile-ready but need eyes
        elif final_decision == "fail":
            delivery_status = "failed"
            is_mobile_ready = False
        
        # Mobile Ready Gate
        if profile_name in ["mobile_preview", "mobile_high"]:
            if is_mobile_ready and poly_count > 150_000:
                # Extra safety, though rules should have caught it
                is_mobile_ready = False
                if final_decision == "pass": final_decision = "review"
                warning_checks.append("mobile_limit_gate")

        # Scoring
        initial_faces = max(int(iso_stats.get("initial_faces", 1)), 1)
        final_faces = int(iso_stats.get("final_faces", 0))
        largest_component_share = float(final_faces / initial_faces)
        
        contamination_score = (
            (1.0 - largest_component_share) * 0.45
            + float(iso_stats.get("removed_plane_face_share", 0.0)) * 0.25
        )
        
        material_grade = self._calculate_material_grade(asset_data)

        # Merge for contamination_report
        combined_report = {}
        combined_report.update(contamination_decisions)
        combined_report.update(texture_integrity_decisions)
        combined_report.update(delivery_decisions)
        combined_report.update(accessor_decisions)
        combined_report["decimation"] = decimation_decision

        return ValidationReport(
            asset_id=asset_id,
            poly_count=poly_count,
            texture_status=asset_data.get("texture_integrity_status", "missing"),
            bbox_reasonable=(bbox_decision == "pass"),
            ground_aligned=(ground_decision == "pass"),
            component_count=int(iso_stats.get("component_count", 1)),
            largest_component_share=largest_component_share,
            contamination_score=contamination_score,
            contamination_report=combined_report,
            mobile_performance_grade=self._calculate_grade(poly_count),
            material_quality_grade=material_grade,
            material_semantic_status=semantic_status,
            
            blocking_checks=blocking_checks,
            warning_checks=warning_checks,
            passed_checks=passed_checks,
            raw_metrics=asset_data,

            texture_quality_status=texture_quality_stats.get("texture_quality_status", "unknown"),
            texture_quality_grade=texture_quality_stats.get("texture_quality_grade", "F"),
            texture_quality_reasons=texture_quality_stats.get("texture_quality_reasons", []),
            black_pixel_ratio=texture_quality_stats.get("black_pixel_ratio", 0.0),
            near_black_ratio=texture_quality_stats.get("near_black_ratio", 0.0),
            dominant_color_ratio=texture_quality_stats.get("dominant_color_ratio", 0.0),
            dominant_background_color_ratio=texture_quality_stats.get("dominant_background_color_ratio", 0.0),
            atlas_coverage_ratio=texture_quality_stats.get("atlas_coverage_ratio", 0.0),
            default_fill_or_flat_color_ratio=texture_quality_stats.get("default_fill_or_flat_color_ratio", 0.0),
            alpha_empty_ratio=texture_quality_stats.get("alpha_empty_ratio", 0.0),
            
            final_decision=final_decision,
            is_mobile_ready=is_mobile_ready,
            delivery_status=delivery_status,
        )

    def _calculate_grade(self, poly_count: int) -> str:
        if poly_count <= 25_000: return "A"
        if poly_count <= 50_000: return "B"
        if poly_count <= 150_000: return "C"
        return "D"

    def _calculate_material_grade(self, asset_data: Dict[str, Any]) -> str:
        status = str(asset_data.get("material_semantic_status", "geometry_only")).lower()
        if status == "pbr_complete": return "S"
        if status == "pbr_partial": return "A"
        if status == "diffuse_textured": return "B"
        if status == "uv_only": return "C"
        return "F"
