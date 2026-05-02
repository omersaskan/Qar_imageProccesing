"""Build and persist the depth_studio manifest block."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


def build_manifest(
    session_id: str,
    input_type: str,
    input_path: str,
    provider: str,
    model_name: Optional[str],
    provider_status: str,
    license_note: str,
    selected_frame_path: Optional[str],
    depth_map_path: Optional[str],
    depth_format: Optional[str],
    refinement_applied: bool,
    mesh_mode: str,
    mesh_vertex_count: int,
    mesh_face_count: int,
    glb_path: Optional[str],
    status: str,
    warnings: List[str],
    enabled: bool = True,
    mask_method: Optional[str] = None,
    mask_fg_ratio: Optional[float] = None,
    mask_bbox: Optional[list] = None,
    mask_full_frame_fallback: bool = False,
    mask_overlay_path: Optional[str] = None,
    mask_stats_path: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "enabled": enabled,
        "mode": "depth_studio",
        "session_id": session_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),

        # Provider
        "provider": provider,
        "model_name": model_name,
        "provider_status": provider_status,
        "license_note": license_note,

        # Input
        "input_type": input_type,
        "input_path": input_path,
        "selected_frame_path": selected_frame_path,

        # Depth
        "depth_map_path": depth_map_path,
        "depth_format": depth_format,
        "refinement_applied": refinement_applied,

        # Mesh
        "mesh_mode": mesh_mode,
        "mesh_vertex_count": mesh_vertex_count,
        "mesh_face_count": mesh_face_count,

        # Masking
        "mask_method": mask_method,
        "mask_fg_ratio": mask_fg_ratio,
        "mask_bbox": mask_bbox,
        "mask_full_frame_fallback": mask_full_frame_fallback,
        "mask_overlay_path": mask_overlay_path,
        "mask_stats_path": mask_stats_path,

        # Output
        "glb_path": glb_path,
        "status": status,
        "warnings": warnings,

        # Asset class metadata
        "is_true_3d": False,
        "has_backside": False,
        "preview_only": True,
        "explicit_final_override_required": True,
    }


def write_manifest(manifest: Dict[str, Any], output_dir: str) -> str:
    """Write depth_studio_manifest.json under output_dir. Returns path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(output_dir) / "depth_studio_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return str(out)
