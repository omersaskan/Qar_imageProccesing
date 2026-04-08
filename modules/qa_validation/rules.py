from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ValidationThresholds(BaseModel):
    polycount_pass: int = Field(50_000, ge=0)
    polycount_review: int = Field(100_000, ge=0)
    bbox_max_dimension_cm: float = Field(500.0, ge=0)
    ground_alignment_threshold_cm: float = Field(1.0, ge=0)
    
    # Contamination Thresholds
    min_largest_component_share_pass: float = Field(0.85, ge=0, le=1.0)
    min_largest_component_share_review: float = Field(0.70, ge=0, le=1.0)
    max_component_count: int = Field(5, ge=1)
    max_plane_face_share: float = Field(0.2, ge=0, le=1.0)
    max_plane_vertex_ratio: float = Field(0.3, ge=0, le=1.0)
    
    # Phase 3: New Robust Thresholds
    min_compactness: float = Field(0.01, ge=0) # Products shouldn't be too sparse
    min_flatness: float = Field(0.02, ge=0)    # Products shouldn't be paper thin slabs
    max_bbox_aspect_ratio: float = Field(20.0, ge=0) # Sanity check for extremely long assets

def validate_polycount(count: int, thresholds: ValidationThresholds) -> str:
    if count <= thresholds.polycount_pass:
        return "pass"
    elif count <= thresholds.polycount_review:
        return "review"
    return "fail"

def validate_contamination(stats: Dict[str, Any], thresholds: ValidationThresholds) -> Dict[str, str]:
    """
    Evaluates scene contamination based on component and plane metrics.
    """
    results = {}
    
    # 1. Largest Component Share
    iso_stats = stats.get("isolation", {})
    final_f = iso_stats.get("final_faces", 0)
    initial_f = iso_stats.get("initial_faces", 1)
    share = final_f / initial_f if initial_f > 0 else 0
    
    if share >= thresholds.min_largest_component_share_pass:
        results["component_share"] = "pass"
    elif share >= thresholds.min_largest_component_share_review:
        results["component_share"] = "review"
    else:
        results["component_share"] = "fail"

    # 2. Component Count
    comp_count = iso_stats.get("component_count", 1)
    if comp_count == 1:
        results["component_count"] = "pass"
    elif comp_count <= thresholds.max_component_count:
        results["component_count"] = "review"
    else:
        # Extreme fragmentation is usually a failure
        results["component_count"] = "fail"

    # 3. Plane Contamination
    plane_share = iso_stats.get("removed_plane_face_share", 0.0)
    plane_v_ratio = iso_stats.get("removed_plane_vertex_ratio", 0.0)
    
    if plane_share <= thresholds.max_plane_face_share and plane_v_ratio <= thresholds.max_plane_vertex_ratio:
        results["plane_contamination"] = "pass"
    elif plane_share <= thresholds.max_plane_face_share * 1.5 or plane_v_ratio <= thresholds.max_plane_vertex_ratio * 1.5:
        results["plane_contamination"] = "review"
    else:
        results["plane_contamination"] = "fail"

    # 4. Phase 3: Robustness Checks
    compactness = iso_stats.get("compactness_score", 1.0)
    flatness = iso_stats.get("flatness_score", 1.0)
    
    if compactness < thresholds.min_compactness:
        results["compactness"] = "review"
    else:
        results["compactness"] = "pass"
        
    if flatness < thresholds.min_flatness:
        results["flatness"] = "review"
    else:
        results["flatness"] = "pass"

    return results

def validate_texture(status: str, has_uv: bool = True, has_texture: bool = True) -> str:
    status = status.lower()
    
    # 1. Texture exists but NO UVs
    if has_texture and not has_uv:
        return "fail" # Mesh cannot render the texture
        
    # 2. Status based check
    if status == "complete":
        return "pass"
    elif status == "missing_uv":
        return "fail"
    elif status == "minor_missing":
        return "review"
        
    return "fail"

def validate_bbox(dimensions: Dict[str, float], thresholds: ValidationThresholds) -> str:
    # Simulating sanity check: dimensions shouldn't be zero or excessively large
    for dim, val in dimensions.items():
        if val <= 0:
            return "fail"
        if val > thresholds.bbox_max_dimension_cm:
            return "review"
    return "pass"

def validate_ground_alignment(offset: float, thresholds: ValidationThresholds) -> str:
    if abs(offset) <= thresholds.ground_alignment_threshold_cm:
        return "pass"
    return "review" # Usually reviewable rather than instant fail unless huge
