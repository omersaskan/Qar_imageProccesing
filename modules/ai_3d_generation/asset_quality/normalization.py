"""
Scene normalization audit for GLB files.

analyze_normalization(glb_path) -> dict

Pure analysis only — no destructive transformation.
Uses trimesh when available; degrades gracefully without it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_SCALE_MIN = 0.001
_SCALE_MAX = 1000.0
_GROUND_THRESHOLD = 0.05   # within 5cm of Y=0 is "likely on ground"
_CENTER_DIST_WARN = 1.0    # warn if model center > 1 unit from origin


def analyze_normalization(glb_path: Optional[str]) -> Dict[str, Any]:
    """
    Analyze normalization characteristics of a GLB without modifying it.

    Returns
    -------
    dict with keys:
        enabled, available, applied, analysis, issues, warnings, recommendations
    """
    result: Dict[str, Any] = {
        "enabled": True,
        "available": False,
        "applied": False,
        "analysis": {
            "bounds": None,
            "dimensions": None,
            "center": None,
            "ground_offset": None,
            "largest_axis": None,
            "likely_flat_on_ground": None,
        },
        "issues": [],
        "warnings": [],
        "recommendations": [],
    }

    if not glb_path or not Path(glb_path).exists():
        result["issues"].append("glb_missing")
        return result

    try:
        import trimesh
        import numpy as np

        scene = trimesh.load(glb_path, force="scene")

        all_vertices = []
        if isinstance(scene, trimesh.Scene):
            for mesh in scene.geometry.values():
                if hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
                    all_vertices.append(mesh.vertices)
        elif hasattr(scene, "vertices") and len(scene.vertices) > 0:
            all_vertices.append(scene.vertices)

        result["available"] = True

        if not all_vertices:
            result["issues"].append("no_vertices_found")
            return result

        verts = np.concatenate(all_vertices, axis=0)
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        dims = maxs - mins
        center = (mins + maxs) / 2.0

        axis_names = ["x", "y", "z"]
        largest_axis = axis_names[int(np.argmax(dims))]
        largest_dim = float(dims.max())

        # Y-up convention: bottom of model is mins[1]
        ground_offset = float(mins[1])
        likely_flat = abs(ground_offset) < _GROUND_THRESHOLD

        center_dist = float(np.linalg.norm(center))

        if largest_dim < _SCALE_MIN:
            result["issues"].append("model_too_small")
            result["recommendations"].append(
                "Model scale is extremely small — check units."
            )
        elif largest_dim > _SCALE_MAX:
            result["issues"].append("model_too_large")
            result["recommendations"].append(
                "Model scale is extremely large — check units."
            )

        if center_dist > _CENTER_DIST_WARN:
            result["warnings"].append("model_not_centered")
            result["recommendations"].append(
                f"Model center is {center_dist:.2f} units from origin. "
                "Consider centering for AR/web delivery."
            )

        if not likely_flat:
            result["warnings"].append("ground_alignment_uncertain")
            result["recommendations"].append(
                f"Model bottom is {ground_offset:.3f} units from Y=0. "
                "Consider aligning to ground plane."
            )

        result["analysis"] = {
            "bounds": [
                [round(float(mins[0]), 4), round(float(mins[1]), 4), round(float(mins[2]), 4)],
                [round(float(maxs[0]), 4), round(float(maxs[1]), 4), round(float(maxs[2]), 4)],
            ],
            "dimensions": {
                "x": round(float(dims[0]), 4),
                "y": round(float(dims[1]), 4),
                "z": round(float(dims[2]), 4),
            },
            "center": [
                round(float(center[0]), 4),
                round(float(center[1]), 4),
                round(float(center[2]), 4),
            ],
            "ground_offset": round(ground_offset, 4),
            "largest_axis": largest_axis,
            "likely_flat_on_ground": likely_flat,
        }

    except ImportError:
        result["available"] = True
        result["warnings"].append("trimesh_unavailable")
    except Exception as exc:
        log.warning("normalization audit failed: %s", exc)
        result["available"] = True
        result["warnings"].append("normalization_analysis_failed")

    return result
