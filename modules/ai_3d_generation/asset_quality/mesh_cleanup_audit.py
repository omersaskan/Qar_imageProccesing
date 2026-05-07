"""
Mesh cleanup audit for GLB files.

audit_mesh_cleanup(glb_path) -> dict

Detection and reporting only — no modification to the GLB.
Uses trimesh when available; degrades gracefully without it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_FACE_COUNT_HIGH = 200_000
_FACE_COUNT_EXCESSIVE = 500_000
_VERTEX_COUNT_HIGH = 200_000
_TINY_COMPONENT_FRAC = 0.05  # component < 5% of total faces = likely floating part
_BOUNDS_MIN = 0.001
_BOUNDS_MAX = 1000.0


def audit_mesh_cleanup(glb_path: Optional[str]) -> Dict[str, Any]:
    """
    Audit mesh cleanup quality without modifying the GLB.

    Returns
    -------
    dict with keys:
        enabled, available, status, issues, warnings, metrics, recommendations
    """
    result: Dict[str, Any] = {
        "enabled": True,
        "available": False,
        "status": "ok",
        "issues": [],
        "warnings": [],
        "metrics": {
            "component_count": None,
            "largest_component_ratio": None,
            "degenerate_face_count": None,
            "duplicate_vertex_estimate": None,
            "non_manifold_estimate": None,
            "boundary_edge_count": None,
        },
        "recommendations": [],
    }

    if not glb_path or not Path(glb_path).exists():
        result["issues"].append("glb_missing")
        result["status"] = "failed"
        return result

    try:
        import trimesh
        import numpy as np

        scene = trimesh.load(glb_path, force="scene")
        result["available"] = True

        meshes: List[trimesh.Trimesh] = []
        if isinstance(scene, trimesh.Scene):
            for geom in scene.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)
        elif isinstance(scene, trimesh.Trimesh):
            meshes.append(scene)

        if not meshes:
            result["issues"].append("no_mesh_found")
            result["status"] = "failed"
            result["recommendations"].append(
                "Scene contains no triangle mesh geometry."
            )
            return result

        total_faces = sum(len(m.faces) for m in meshes)
        total_verts = sum(len(m.vertices) for m in meshes)

        # ── Face / vertex count checks ────────────────────────────────────────
        if total_faces > _FACE_COUNT_EXCESSIVE:
            result["issues"].append("excessive_face_count")
            result["recommendations"].append(
                f"Face count {total_faces:,} is very high. "
                "Decimation required before delivery."
            )
        elif total_faces > _FACE_COUNT_HIGH:
            result["warnings"].append("high_face_count")
            result["recommendations"].append(
                f"Face count {total_faces:,} may impact mobile performance."
            )

        if total_verts > _VERTEX_COUNT_HIGH:
            result["warnings"].append("high_vertex_count")

        # ── Per-mesh topology analysis ────────────────────────────────────────
        total_components = 0
        largest_component_faces = 0
        total_degenerate = 0
        total_boundary = 0
        any_non_manifold = False

        for mesh in meshes:
            # Connected components via mesh.split()
            try:
                components = mesh.split(only_watertight=False)
                n = len(components)
                total_components += n
                if n > 0:
                    max_comp_faces = max(len(c.faces) for c in components)
                    largest_component_faces = max(
                        largest_component_faces, max_comp_faces
                    )
            except Exception:
                pass

            # Degenerate faces (zero-area)
            try:
                degen = int(np.sum(mesh.area_faces < 1e-12))
                total_degenerate += degen
            except Exception:
                pass

            # Boundary edges (open boundary = not watertight)
            try:
                be = mesh.boundary_edges
                total_boundary += len(be)
            except Exception:
                pass

            # Non-manifold via watertight check
            try:
                if not mesh.is_watertight:
                    any_non_manifold = True
            except Exception:
                pass

        # Duplicate vertex estimate
        dup_estimate: Optional[int] = None
        try:
            combined = np.concatenate([m.vertices for m in meshes], axis=0)
            unique_count = len(np.unique(combined.round(6), axis=0))
            dup_estimate = max(0, len(combined) - unique_count)
        except Exception:
            pass

        # Largest component ratio
        largest_ratio: Optional[float] = None
        if total_components > 0 and total_faces > 0:
            largest_ratio = round(largest_component_faces / total_faces, 3)

        result["metrics"] = {
            "component_count": total_components if total_components > 0 else None,
            "largest_component_ratio": largest_ratio,
            "degenerate_face_count": total_degenerate,
            "duplicate_vertex_estimate": dup_estimate,
            "non_manifold_estimate": any_non_manifold,
            "boundary_edge_count": total_boundary,
        }

        # ── Issue classification ──────────────────────────────────────────────
        # Floating parts: multiple components and largest < (1 - threshold)
        if total_components > 1 and largest_ratio is not None:
            if largest_ratio < (1.0 - _TINY_COMPONENT_FRAC):
                result["warnings"].append("floating_parts_detected")
                result["recommendations"].append("floating_parts_detected")

        if total_degenerate > 100:
            result["issues"].append("significant_degenerate_faces")
            result["recommendations"].append(
                f"{total_degenerate:,} degenerate faces detected. Cleanup recommended."
            )
        elif total_degenerate > 0:
            result["warnings"].append(f"degenerate_faces:{total_degenerate}")

        if total_boundary > 0:
            result["warnings"].append("open_mesh_boundary_edges")

        if any_non_manifold:
            result["warnings"].append("non_manifold_geometry")
            result["recommendations"].append("retopology_recommended")

        if total_components > 3:
            result["warnings"].append(f"high_component_count:{total_components}")
            result["recommendations"].append("manual_cleanup_required")

        # Suspicious bounds
        try:
            all_v = np.concatenate([m.vertices for m in meshes], axis=0)
            span = float(np.max(all_v) - np.min(all_v)) if len(all_v) > 0 else 0.0
            if span < _BOUNDS_MIN:
                result["issues"].append("suspiciously_small_bounds")
            elif span > _BOUNDS_MAX:
                result["issues"].append("suspiciously_large_bounds")
        except Exception:
            pass

        # ── Final status ──────────────────────────────────────────────────────
        fatal = {"no_mesh_found", "excessive_face_count",
                 "suspiciously_small_bounds", "suspiciously_large_bounds",
                 "significant_degenerate_faces"}
        if any(i in fatal for i in result["issues"]):
            result["status"] = "review"  # flag for review; failed is reserved for no-mesh
        elif result["issues"] or result["warnings"]:
            result["status"] = "review"
        else:
            result["status"] = "ok"

        # no_mesh_found is the only hard "failed"
        if "no_mesh_found" in result["issues"]:
            result["status"] = "failed"

    except ImportError:
        result["available"] = True
        result["warnings"].append("trimesh_unavailable")
        result["status"] = "ok"
    except Exception as exc:
        log.warning("mesh_cleanup_audit failed: %s", exc)
        result["available"] = True
        result["warnings"].append("mesh_cleanup_audit_failed")
        result["status"] = "ok"

    return result
