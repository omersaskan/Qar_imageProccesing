"""Assemble final preview GLB from depth mesh + texture."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from .depth_output import load_depth_png16
from .depth_to_mesh import depth_to_glb
from modules.operations.settings import settings


def build_glb(
    depth_map_path: str,
    texture_path: str,
    output_glb_path: str,
    grid_resolution: int = None,
    depth_scale: float = 0.3,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Load depth map, build displaced plane mesh, texture it, export GLB.
    Returns manifest-ready dict.
    """
    res = grid_resolution or settings.depth_grid_resolution

    try:
        depth = load_depth_png16(depth_map_path)
    except Exception as e:
        return {"status": "failed", "reason": f"Cannot load depth map: {e}", "glb_path": None,
                "mesh_vertex_count": 0, "mesh_face_count": 0}

    if not Path(texture_path).exists():
        return {"status": "failed", "reason": f"Texture not found: {texture_path}", "glb_path": None,
                "mesh_vertex_count": 0, "mesh_face_count": 0}

    return depth_to_glb(
        depth=depth,
        texture_image_path=texture_path,
        output_glb_path=output_glb_path,
        grid_resolution=res,
        depth_scale=depth_scale,
        mask=mask,
    )
