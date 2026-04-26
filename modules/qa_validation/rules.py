from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ValidationThresholds(BaseModel):
    # geometry / runtime thresholds
    # Profile limits are now handled dynamically, but these remain as global defaults
    polycount_pass: int = Field(50_000, ge=0)
    polycount_review: int = Field(150_000, ge=0)
    polycount_fail: int = Field(250_000, ge=0)
    
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

    # texture quality thresholds
    max_black_pixel_ratio: float = Field(0.40, ge=0, le=1.0)
    max_near_black_ratio: float = Field(0.60, ge=0, le=1.0)
    max_flat_color_ratio: float = Field(0.70, ge=0, le=1.0)
    max_dominant_background_ratio: float = Field(0.50, ge=0, le=1.0)
    min_atlas_coverage_ratio: float = Field(0.30, ge=0, le=1.0) # Tightened from 0.05
    min_near_white_ratio_white_cream: float = Field(0.40, ge=0, le=1.0)


def validate_polycount_by_profile(count: int, profile_name: str) -> str:
    """
    Hard gates for delivery profiles:
    - mobile_preview > 50k => fail
    - mobile_high > 150k => fail
    - desktop_high > 250k => fail
    - raw_archive => always pass (but not mobile ready)
    """
    p = profile_name.lower()
    if p == "mobile_preview":
        if count > 50_000: return "fail"
        if count > 45_000: return "review"
    elif p == "mobile_high":
        if count > 150_000: return "fail"
        if count > 130_000: return "review"
    elif p == "desktop_high":
        if count > 250_000: return "fail"
        if count > 220_000: return "review"
    elif p == "raw_archive":
        return "pass"
    
    # Standard fallback
    if count > 150_000: return "review"
    return "pass"


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
    texture_applied = bool(asset_data.get("texture_applied", False))

    # SPRINT 5: Rigid integrity fix
    # if texture_count=0/material_count=0/has_uv=false, texture_status must not be "complete"
    if status == "complete":
        if not (has_uv and has_material and texture_count > 0):
             status = "geometry_only" if not has_uv else "missing"

    # 1. Core Integrity (did it survive?)
    if status == "complete":
        if texture_applied and texture_count > 0 and has_uv and has_material:
            results["uv_integrity"] = "pass"
            results["application"] = "pass"
            results["material_integrity"] = "pass"
        else:
            results["uv_integrity"] = "pass" if has_uv else "fail"
            results["application"] = "pass" if (texture_applied and texture_count > 0) else "fail"
            results["material_integrity"] = "pass" if has_material else "fail"
    elif status == "degraded":
        results["uv_integrity"] = "pass" if has_uv else "review"
        results["application"] = "pass" if texture_count > 0 else "review"
        results["material_integrity"] = "pass" if has_material else "review"
    else:
        results["uv_integrity"] = "fail" if not has_uv else "pass" # If geometry_only, pass UV check? 
        # Actually, let's keep it simple:
        results["uv_integrity"] = "pass" if has_uv else "fail"
        results["application"] = "pass" if texture_count > 0 else "fail"
        results["material_integrity"] = "pass" if has_material else "fail"
        
    # 2. Semantic Richness (how good is it?)
    results["material_semantics"] = validate_material_semantics(sem_status)
    results["final_texture_status"] = status # For reporting

    return results


def validate_material_semantics(status: str) -> str:
    status = status.lower()
    if status in {"pbr_complete", "pbr_partial", "diffuse_textured"}:
        return "pass"
    if status in {"uv_only", "material_incomplete"}:
        return "review"
    return "fail"


def validate_bbox(dimensions: Dict[str, float], thresholds: ValidationThresholds) -> str:
    if not dimensions:
        return "fail"
    for _, val in dimensions.items():
        if val <= 0: return "fail"
        if val > thresholds.bbox_max_dimension_cm: return "review"
    return "pass"


def validate_ground_alignment(offset: float, thresholds: ValidationThresholds) -> str:
    if abs(offset) <= thresholds.ground_alignment_threshold_cm:
        return "pass"
    return "review"


def validate_contamination(stats: Dict[str, Any], thresholds: ValidationThresholds) -> Dict[str, str]:
    results: Dict[str, str] = {}
    iso = stats.get("isolation", {})
    if not iso: return {}

    final_faces = int(iso.get("final_faces", 0))
    initial_faces = max(int(iso.get("initial_faces", 1)), 1)
    share = final_faces / initial_faces

    if share >= thresholds.min_largest_component_share_pass:
        results["component_share"] = "pass"
    elif share >= thresholds.min_largest_component_share_review:
        results["component_share"] = "review"
    else:
        results["component_share"] = "fail"

    comp_count = int(iso.get("component_count", 1))
    if comp_count == 1:
        results["component_count"] = "pass"
    elif comp_count <= thresholds.max_component_count:
        results["component_count"] = "review"
    else:
        results["component_count"] = "fail"

    plane_share = float(iso.get("removed_plane_face_share", 0.0))
    plane_v_ratio = float(iso.get("removed_plane_vertex_ratio", 0.0))
    if plane_share <= thresholds.max_plane_face_share and plane_v_ratio <= thresholds.max_plane_vertex_ratio:
        results["plane_contamination"] = "pass"
    else:
        results["plane_contamination"] = "review"

    compactness = float(iso.get("compactness_score", 0.0))
    results["compactness"] = "pass" if compactness >= thresholds.min_compactness_score else "review"

    selected_score = float(iso.get("selected_component_score", 0.0))
    results["selection_quality"] = "pass" if selected_score >= thresholds.min_selected_component_score else "review"

    return results


def validate_accessors(asset_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Enforces existence of critical GLB accessors (POSITION, NORMAL, TEXCOORD_0).
    """
    results: Dict[str, str] = {}
    
    all_pos = bool(asset_data.get("all_primitives_have_position", False))
    all_norm = bool(asset_data.get("all_primitives_have_normal", False))
    all_uv = bool(asset_data.get("all_textured_primitives_have_texcoord_0", False))
    
    results["accessor_position"] = "pass" if all_pos else "fail"
    results["accessor_normal"] = "pass" if all_norm else "fail"
    
    # UV is mandatory if textured
    sem_status = str(asset_data.get("material_semantic_status", "geometry_only")).lower()
    if sem_status != "geometry_only":
        results["accessor_uv"] = "pass" if all_uv else "fail"
    else:
        results["accessor_uv"] = "pass"
        
    return results


def validate_texture_quality(quality_data: Dict[str, Any]) -> str:
    """
    Hard gate for texture atlas quality.
    """
    status = str(quality_data.get("texture_quality_status", "fail")).lower()
    if status == "success":
        return "pass"
    if status == "review":
        return "review"
    return "fail"


def validate_decimation(stats: Dict[str, Any]) -> str:
    """
    Ensures decimation didn't break UVs/Materials.
    """
    status = str(stats.get("decimation_status", "none")).lower()
    if status == "failed_visual_integrity":
        return "fail"
    if "error" in status:
        return "review"
    
    if not stats.get("uv_preserved", True) or not stats.get("material_preserved", True):
        return "fail"
        
    return "pass"


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


def validate_export_delivery_status(asset_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Validates export_status and delivery_ready flags from the exporter.
    """
    results: Dict[str, str] = {}
    export_status = str(asset_data.get("export_status", "unknown")).lower()
    delivery_ready = bool(asset_data.get("delivery_ready", False))
    profile = str(asset_data.get("delivery_profile", "raw_archive")).lower()
    
    if export_status in ["success", "unknown"]:
        results["export_status"] = "pass"
    elif export_status == "failed_texture_application":
        results["export_status"] = "fail"
    elif export_status == "failed_validation":
        results["export_status"] = "fail"
    else:
        results["export_status"] = "review"
        
    # mobile profiles must be delivery_ready
    if profile in ["mobile_preview", "mobile_high", "desktop_high"]:
        results["export_delivery_gate"] = "pass" if delivery_ready else "fail"
    else:
        results["export_delivery_gate"] = "pass"
        
    return results


def validate_object_filtering(asset_data: Dict[str, Any]) -> str:
    status = str(asset_data.get("filtering_status", "unknown")).lower()
    if status == "object_isolated": return "pass"
    if status == "failed": return "fail"
    if status == "scene_raw": return "review"
    return "review"
