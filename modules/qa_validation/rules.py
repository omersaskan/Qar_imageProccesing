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

    # texture / UV thresholds
    require_uv_for_texture_pass: bool = True

    # delivered artifact thresholds
    max_delivery_component_count_pass: int = Field(2, ge=1)
    max_delivery_component_count_review: int = Field(5, ge=1)


def validate_polycount(count: int, thresholds: ValidationThresholds) -> str:
    if count <= thresholds.polycount_pass:
        return "pass"
    if count <= thresholds.polycount_review:
        return "review"
    return "fail"


def validate_texture(status: str) -> str:
    """
    status expected:
    - complete
    - degraded
    - missing
    """
    status = status.lower()
    if status == "complete":
        return "pass"
    if status in {"degraded", "minor_missing"}:
        return "review"
    return "fail"


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
    """
    results: Dict[str, str] = {}
    iso = stats.get("isolation", {})

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
    Checks:
    - texture exists but no UV
    - texture exists but application failed
    - texture exists but material missing
    """
    results: Dict[str, str] = {}

    texture_path_exists = bool(asset_data.get("texture_path_exists", False))
    has_uv = bool(asset_data.get("has_uv", False))
    has_material = bool(asset_data.get("has_material", False))
    texture_applied_successfully = bool(asset_data.get("texture_applied_successfully", False))

    # texture + UV integrity
    if texture_path_exists and not has_uv and thresholds.require_uv_for_texture_pass:
        results["texture_uv_integrity"] = "review"
    elif texture_path_exists and has_uv:
        results["texture_uv_integrity"] = "pass"
    else:
        results["texture_uv_integrity"] = "pass" if not texture_path_exists else "review"

    # texture application
    if texture_path_exists and not texture_applied_successfully:
        results["texture_application"] = "review"
    else:
        results["texture_application"] = "pass"

    # material integrity
    if texture_path_exists and not has_material:
        results["material_integrity"] = "review"
    else:
        results["material_integrity"] = "pass"

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
