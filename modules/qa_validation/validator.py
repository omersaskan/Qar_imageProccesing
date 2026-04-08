from typing import Dict, Any, List
from modules.shared_contracts.models import ValidationReport
from .rules import (
    ValidationThresholds, 
    validate_polycount, 
    validate_texture, 
    validate_bbox, 
    validate_ground_alignment
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
        texture_decision = validate_texture(asset_data.get("texture_status", "unknown"))
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
        
        # A simple contamination score: 1.0 - share
        contam_score = 1.0 - share

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
            final_decision=final_decision
        )

    def _calculate_grade(self, poly_count: int) -> str:
        if poly_count <= 25_000:
            return "A"
        elif poly_count <= 50_000:
            return "B"
        elif poly_count <= 100_000:
            return "C"
        return "D"
