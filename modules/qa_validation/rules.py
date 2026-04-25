from typing import Dict, Any
from pydantic import BaseModel, Field


class ValidationThresholds(BaseModel):
    # geometry / runtime thresholds
    polycount_pass: int = Field(50_000, ge=0)
    polycount_review: int = Field(100_000, ge=0)
    bbox_max_dimension_cm: float = Field(500.0, ge=0)
    ground_alignment_threshold_cm: float = Field(1.0, ge=0)

    # contamination thresholds
    min_largest_component_share_pass: float = Field(0.85, ge=0, le=1.0)
    min_largest_component_share_review: float = Field(0.70, ge=0, le=1.0)
    max_component_count: int = Field(5, ge=1)
    max_plane_face_share: float = Field(0.20, ge=0, le=1.0)
    max_plane_vertex_ratio: float = Field(0.30, ge=0, le=1.0)

    # product-likeness thresholds
    min_compactness_score: float = Field(0.01, ge=0, le=1.0)
    min_selected_component_score: float = Field(0.20, ge=0, le=1.0)

    # delivered artifact thresholds
    max_delivery_component_count_pass: int = Field(2, ge=1)
    max_delivery_component_count_review: int = Field(5, ge=1)
    
    # material / semantic requirements
    require_pbr_complete: bool = False


def validate_polycount(count: int, thresholds: ValidationThresholds) -> str:
    if count <= thresholds.polycount_pass:
        return "pass"
    if count <= thresholds.polycount_review:
        return "review"
    return "fail"


def validate_texture(status: str) -> str:
    """
    Integrity focus: whether mapping/packaging survived.
    """
    status = status.lower()
    if status == "complete":
        return "pass"
    if status in {"degraded", "minor_missing"}:
        return "review"
    if status == "missing":
        return "fail"
    return "fail"


def validate_material_semantics(status: str) -> str:
    """
    Richness focus: diffuse vs partial PBR vs complete PBR.
    - geometry_only -> fail
    - uv_only -> review
    - diffuse_textured -> pass
    - pbr_partial -> pass
    - pbr_complete -> pass
    """
    status = status.lower()
    if status in {"pbr_complete", "pbr_partial", "diffuse_textured"}:
        return "pass"
    if status in {"uv_only", "material_incomplete"}:
        return "review"
    return "fail" # geometry_only or unknown



def validate_bbox(dimensions: Dict[str, float], thresholds: ValidationThresholds) -> str:
    if not dimensions:
        return "fail"

    for _, val in dimensions.items():
        if val <= 0:
            return "fail"
        if val > thresholds.bbox_max_dimension_cm:
            return "review"

    return "pass"


def validate_ground_alignment(offset: float, thresholds: ValidationThresholds) -> str:
    if abs(offset) <= thresholds.ground_alignment_threshold_cm:
        return "pass"
    return "review"


def validate_contamination(stats: Dict[str, Any], thresholds: ValidationThresholds) -> Dict[str, str]:
    """
    Uses cleanup_stats -> isolation stats to assess scene contamination.
    If no isolation stats are present (e.g. standalone validation without a
    cleanup pipeline), returns all-pass so caller judgment is not corrupted.
    """
    results: Dict[str, str] = {}
    iso = stats.get("isolation", {})

    # If there is no isolation data at all, assume clean (caller didn't run cleanup)
    if not iso:
        return {}

    final_faces = int(iso.get("final_faces", 0))
    initial_faces = max(int(iso.get("initial_faces", 1)), 1)
    share = final_faces / initial_faces

    # 1) largest component share
    if share >= thresholds.min_largest_component_share_pass:
        results["component_share"] = "pass"
    elif share >= thresholds.min_largest_component_share_review:
        results["component_share"] = "review"
    else:
        results["component_share"] = "fail"

    # 2) component count
    comp_count = int(iso.get("component_count", 1))
    if comp_count == 1:
        results["component_count"] = "pass"
    elif comp_count <= thresholds.max_component_count:
        results["component_count"] = "review"
    else:
        results["component_count"] = "fail"

    # 3) plane contamination
    plane_share = float(iso.get("removed_plane_face_share", 0.0))
    plane_v_ratio = float(iso.get("removed_plane_vertex_ratio", 0.0))

    if plane_share <= thresholds.max_plane_face_share and plane_v_ratio <= thresholds.max_plane_vertex_ratio:
        results["plane_contamination"] = "pass"
    elif (
        plane_share <= thresholds.max_plane_face_share * 1.5
        and plane_v_ratio <= thresholds.max_plane_vertex_ratio * 1.5
    ):
        results["plane_contamination"] = "review"
    else:
        results["plane_contamination"] = "fail"

    # 4) compactness
    compactness = float(iso.get("compactness_score", 0.0))
    if compactness >= thresholds.min_compactness_score:
        results["compactness"] = "pass"
    else:
        results["compactness"] = "review"

    # 5) selected component score
    selected_score = float(iso.get("selected_component_score", 0.0))
    if selected_score >= thresholds.min_selected_component_score:
        results["selection_quality"] = "pass"
    else:
        results["selection_quality"] = "review"

    return results


def validate_texture_integrity(asset_data: Dict[str, Any], thresholds: ValidationThresholds) -> Dict[str, str]:
    """
    Honest checks based entirely on the delivered GLB stats.
    Uses: texture_integrity_status, has_uv, has_material, texture_count.
    """
    results: Dict[str, str] = {}
    
    status = str(asset_data.get("texture_integrity_status", "missing")).lower()
    sem_status = str(asset_data.get("material_semantic_status", "geometry_only")).lower()
    has_uv = bool(asset_data.get("has_uv", False))
    has_material = bool(asset_data.get("has_material", False))
    texture_count = int(asset_data.get("texture_count", 0))

    # 1. Core Integrity (did it survive?)
    if status == "complete":
        results["texture_uv_integrity"] = "pass"
        results["texture_application"] = "pass"
        results["material_integrity"] = "pass"
    elif status == "degraded":
        results["texture_uv_integrity"] = "pass" if has_uv else "review"
        results["texture_application"] = "pass" if texture_count > 0 else "review"
        results["material_integrity"] = "pass" if has_material else "review"
    else:
        results["texture_uv_integrity"] = "fail"
        results["texture_application"] = "fail"
        results["material_integrity"] = "fail"
        
    # 2. Semantic Richness (how good is it?)
    results["material_semantics"] = validate_material_semantics(sem_status)

    return results


def validate_delivery_mesh(asset_data: Dict[str, Any], thresholds: ValidationThresholds) -> Dict[str, str]:
    results: Dict[str, str] = {}

    if "delivery_geometry_count" in asset_data:
        geometry_count = int(asset_data.get("delivery_geometry_count", 0))
        results["delivery_geometry"] = "pass" if geometry_count > 0 else "fail"

    if "delivery_component_count" in asset_data:
        component_count = int(asset_data.get("delivery_component_count", 0))
        if component_count <= thresholds.max_delivery_component_count_pass:
            results["delivery_fragmentation"] = "pass"
        elif component_count <= thresholds.max_delivery_component_count_review:
            results["delivery_fragmentation"] = "review"
        else:
            results["delivery_fragmentation"] = "fail"

    return results

def validate_texture_quality(quality_data: Dict[str, Any]) -> str:
    """
    Assesses quality status from TextureQualityAnalyzer.
    """
    status = str(quality_data.get("texture_quality_status", "unknown")).lower()
    if status == "clean":
        return "pass"
    if status == "warning":
        return "review"
    if status == "contaminated":
        return "fail"
    return "fail"
