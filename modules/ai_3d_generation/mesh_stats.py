"""
Reusable mesh statistics extractor for GLB files.

Uses trimesh when available; degrades gracefully when not installed.
Never modifies the GLB file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


def extract_mesh_stats(glb_path: Optional[str]) -> Dict[str, Any]:
    """
    Extract vertex/face counts from a GLB file using trimesh.

    Parameters
    ----------
    glb_path : str | None
        Absolute or relative path to the GLB file.

    Returns
    -------
    dict with keys:
        enabled        : True
        available      : bool — True only when counts were successfully read
        vertex_count   : int | None
        face_count     : int | None
        geometry_count : int | None
        error          : None | "glb_missing" | "trimesh_unavailable" | "mesh_stats_failed"
    """
    result: Dict[str, Any] = {
        "enabled": True,
        "available": False,
        "vertex_count": None,
        "face_count": None,
        "geometry_count": None,
        "error": None,
    }

    if not glb_path or not Path(glb_path).exists():
        result["error"] = "glb_missing"
        return result

    try:
        import trimesh

        scene = trimesh.load(glb_path, force="scene")
        if isinstance(scene, trimesh.Scene):
            geometry_count = len(scene.geometry)
            vertex_count = sum(
                len(m.vertices) for m in scene.geometry.values() if hasattr(m, "vertices")
            )
            face_count = sum(
                len(m.faces) for m in scene.geometry.values() if hasattr(m, "faces")
            )
        else:
            geometry_count = 1
            vertex_count = len(scene.vertices)
            face_count = len(scene.faces)

        result["available"] = True
        result["vertex_count"] = vertex_count
        result["face_count"] = face_count
        result["geometry_count"] = geometry_count

    except ImportError:
        result["error"] = "trimesh_unavailable"
    except Exception as exc:
        log.warning("mesh_stats: failed to read %s: %s", glb_path, exc)
        result["error"] = "mesh_stats_failed"

    return result
