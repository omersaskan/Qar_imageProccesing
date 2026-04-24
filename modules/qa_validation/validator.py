from typing import Dict, Any

from modules.shared_contracts.models import ValidationReport
from .rules import (
    ValidationThresholds,
    validate_polycount,
    validate_texture,
    validate_bbox,
    validate_ground_alignment,
    validate_contamination,
    validate_texture_integrity,
    validate_delivery_mesh,
)


class AssetValidator:
    def __init__(self, thresholds: ValidationThresholds = None):
        self.thresholds = thresholds or ValidationThresholds()

    def validate(self, asset_id: str, asset_data: Dict[str, Any]) -> ValidationReport:
        """
        asset_data expected keys:
        - poly_count: int
        - texture_status: str
        - bbox: Dict[str, float]
        - ground_offset: float
        - cleanup_stats: Dict[str, Any]
        - texture_path_exists: bool
        - has_uv: bool
        - has_material: bool
        - has_embedded_texture: bool
        - texture_count: int
        - material_count: int
        - texture_integrity_status: str
        - material_integrity_status: str
        - material_semantic_status: str
        - basecolor_present: bool
        - normal_present: bool
        - metallic_roughness_present: bool
        """

        poly_decision = validate_polycount(asset_data.get("poly_count", 0), self.thresholds)
        texture_status = asset_data.get("texture_integrity_status", "missing")
        semantic_status = asset_data.get("material_semantic_status", "geometry_only")
        
        texture_decision = validate_texture(texture_status)
        bbox_decision = validate_bbox(asset_data.get("bbox", {}), self.thresholds)
        ground_decision = validate_ground_alignment(asset_data.get("ground_offset", 99.0), self.thresholds)

        cleanup_stats = asset_data.get("cleanup_stats", {})
        iso_stats = cleanup_stats.get("isolation", {})

        contamination_decisions = validate_contamination(cleanup_stats, self.thresholds)
        texture_integrity_decisions = validate_texture_integrity(asset_data, self.thresholds)
        delivery_decisions = validate_delivery_mesh(asset_data, self.thresholds)

        all_decisions = [
            poly_decision,
            texture_decision,
            bbox_decision,
            ground_decision,
            *contamination_decisions.values(),
            *texture_integrity_decisions.values(),
            *delivery_decisions.values(),
        ]

        if "fail" in all_decisions:
            final_decision = "fail"
        elif "review" in all_decisions:
            final_decision = "review"
        else:
            final_decision = "pass"

        # SPRINT: Customer-ready Guard
        # Geometry-only or UV-only GLBs must never pass as customer-ready
        if semantic_status in ["geometry_only", "uv_only", "material_incomplete"]:
            final_decision = "fail"
            combined_report = {"semantic_guard": "fail"} # Initialize dict here to be updated below

        comp_count = int(iso_stats.get("component_count", 1))
        final_faces = int(iso_stats.get("final_faces", 0))
        initial_faces = max(int(iso_stats.get("initial_faces", 1)), 1)
        largest_component_share = float(final_faces / initial_faces)

        plane_share = float(iso_stats.get("removed_plane_face_share", 0.0))
        comp_overflow = max(0.0, float(comp_count - self.thresholds.max_component_count)) / 10.0
        texture_penalty = 0.15 if "review" in texture_integrity_decisions.values() else 0.0

        contamination_score = (
            (1.0 - largest_component_share) * 0.45
            + plane_share * 0.25
            + min(0.20, comp_overflow) * 0.15
            + texture_penalty * 0.15
        )
        
        # New: Material Quality Grade
        material_grade = self._calculate_material_grade(asset_data)

        if "combined_report" not in locals():
            combined_report = {}
        combined_report.update(contamination_decisions)
        combined_report.update(texture_integrity_decisions)
        combined_report.update(delivery_decisions)

        return ValidationReport(
            asset_id=asset_id,
            poly_count=asset_data.get("poly_count", 0),
            texture_status=texture_status,
            bbox_reasonable=(bbox_decision == "pass"),
            ground_aligned=(ground_decision == "pass"),
            component_count=comp_count,
            largest_component_share=largest_component_share,
            contamination_score=contamination_score,
            contamination_report=combined_report,
            mobile_performance_grade=self._calculate_grade(asset_data.get("poly_count", 0)),
            material_quality_grade=material_grade,
            material_semantic_status=semantic_status,
            final_decision=final_decision,
        )

    def _calculate_grade(self, poly_count: int) -> str:
        if poly_count <= 25_000:
            return "A"
        if poly_count <= 50_000:
            return "B"
        if poly_count <= 100_000:
            return "C"
        return "D"

    def _calculate_material_grade(self, asset_data: Dict[str, Any]) -> str:
        status = str(asset_data.get("material_semantic_status", "geometry_only")).lower()
        if status == "pbr_complete":
            return "S"  # State of the art PBR
        if status == "pbr_partial":
            return "A"  # High quality but missing minor channels
        if status == "diffuse_textured":
            return "B"  # Valid photogrammetry
        if status == "uv_only":
            return "C"  # Map ready but untextured
        return "F"  # Fail / Geometry only

