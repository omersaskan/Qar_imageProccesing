from typing import Dict, Any, List
from modules.shared_contracts.models import ValidationReport
from .rules import (
    ValidationThresholds,
    validate_polycount, 
    validate_texture, 
    validate_bbox, 
    validate_ground_alignment,
    validate_contamination
)

class AssetValidator:
    def __init__(self, thresholds: ValidationThresholds = None):
        self.thresholds = thresholds or ValidationThresholds()

    def validate(self, asset_id: str, asset_data: Dict[str, Any]) -> ValidationReport:
        """
        Validates asset data against thresholds.
        asset_data keys:
        - poly_count: int
        - texture_status: str
        - bbox: Dict[str, float]
        - ground_offset: float
        - cleanup_stats: Dict[str, Any]
        """
        poly_decision = validate_polycount(asset_data.get("poly_count", 0), self.thresholds)
        
        # Phase 1: Improved texture validation
        texture_decision = validate_texture(
            status=asset_data.get("texture_status", "unknown"),
            has_uv=asset_data.get("has_uv", True),
            has_texture=asset_data.get("has_texture", True)
        )
        
        bbox_decision = validate_bbox(asset_data.get("bbox", {}), self.thresholds)
        ground_decision = validate_ground_alignment(asset_data.get("ground_offset", 99.0), self.thresholds)
        
        # New Contamination Validation
        cleanup_stats = asset_data.get("cleanup_stats", {})
        iso_stats = cleanup_stats.get("isolation", {})
        contamination_decisions = validate_contamination(cleanup_stats, self.thresholds)
        
        # Combine decisions: fail beats all, review beats pass
        all_decisions = [poly_decision, texture_decision, bbox_decision, ground_decision]
        all_decisions.extend(contamination_decisions.values())
        
        if "fail" in all_decisions:
            final_decision = "fail"
        elif "review" in all_decisions:
            final_decision = "review"
        else:
            final_decision = "pass"
            
        # Calculation scores
        comp_count = iso_stats.get("component_count", 1)
        final_f = iso_stats.get("final_faces", 0)
        initial_f = iso_stats.get("initial_faces", 1)
        share = final_f / initial_f if initial_f > 0 else 0
        
        # Weighted contamination score
        # 1. Component share (dominant part of the score)
        # 2. Plane contamination (if isolation stats available)
        # 3. Component count overflow penalty
        
        share_weight = 0.5
        plane_weight = 0.3
        count_weight = 0.2
        
        plane_share = iso_stats.get("removed_plane_face_share", 0.0)
        comp_overflow = max(0, iso_stats.get("component_count", 1) - self.thresholds.max_component_count) / 10.0
        
        contam_score = (
            (1.0 - share) * share_weight +
            plane_share * plane_weight +
            min(0.2, comp_overflow) * count_weight
        )

        return ValidationReport(
            asset_id=asset_id,
            poly_count=asset_data.get("poly_count", 0),
            texture_status=asset_data.get("texture_status", "unknown"),
            bbox_reasonable=(bbox_decision == "pass"),
            ground_aligned=(ground_decision == "pass"),
            component_count=comp_count,
            largest_component_share=share,
            contamination_score=contam_score,
            contamination_report=contamination_decisions,
            mobile_performance_grade=self._calculate_grade(asset_data.get("poly_count", 0)),
            final_decision=final_decision,
            # Phase 3 Fields
            flatness_score=iso_stats.get("flatness_score", 0.0),
            compactness_score=iso_stats.get("compactness_score", 0.0),
            selected_component_score=iso_stats.get("selected_component_score", 0.0)
        )

    def _calculate_grade(self, poly_count: int) -> str:
        if poly_count <= 25_000:
            return "A"
        elif poly_count <= 50_000:
            return "B"
        elif poly_count <= 100_000:
            return "C"
        return "D"
