"""Observed-surface ratio: how much of the mesh is actually backed by capture."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None  # type: ignore


def compute_observed_surface_ratio(
    mesh: "trimesh.Trimesh",
    cameras: Optional[List[Dict]] = None,
    masks: Optional[Dict[str, np.ndarray]] = None,
    point_cloud: Optional["trimesh.points.PointCloud"] = None,
    pc_dist_threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Estimate the fraction of the mesh surface area that is supported by
    actual observation, using whichever signal is available:

        - dense point cloud proximity (best signal)
        - per-view mask projection (fallback)
        - geometric heuristic (last resort: assume fully observed)

    Returns:
        {
          "observed_surface_ratio": float in [0,1],
          "method": "point_cloud" | "mask_projection" | "heuristic",
          "supported_face_count": int,
          "total_face_count": int,
        }
    """
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        return {
            "observed_surface_ratio": 0.0,
            "method": "empty_mesh",
            "supported_face_count": 0,
            "total_face_count": 0,
        }

    total_faces = int(len(mesh.faces))

    # 1. Best signal: dense point cloud — count faces whose centroid is near a point
    if point_cloud is not None and len(getattr(point_cloud, "vertices", [])) > 0:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(point_cloud.vertices)
            face_centers = mesh.triangles_center
            dists, _ = tree.query(face_centers, k=1)
            supported = int(np.sum(dists < pc_dist_threshold))
            ratio = supported / max(total_faces, 1)
            return {
                "observed_surface_ratio": float(ratio),
                "method": "point_cloud",
                "supported_face_count": supported,
                "total_face_count": total_faces,
            }
        except Exception:
            pass

    # 2. Mask projection — leverage existing camera_projection helper if available
    if cameras and masks:
        try:
            from modules.asset_cleanup_pipeline.camera_projection import (
                compute_component_mask_support,
            )
            sup = compute_component_mask_support(mesh, cameras, masks)
            avg = float(sup.get("avg_support", 0.0))
            return {
                "observed_surface_ratio": float(np.clip(avg, 0.0, 1.0)),
                "method": "mask_projection",
                "supported_face_count": int(round(avg * total_faces)),
                "total_face_count": total_faces,
            }
        except Exception:
            pass

    # 3. Heuristic fallback — assume fully observed (conservative for completion gating)
    return {
        "observed_surface_ratio": 1.0,
        "method": "heuristic",
        "supported_face_count": total_faces,
        "total_face_count": total_faces,
    }


def compute_synthesized_ratio(
    original_face_count: int,
    completed_face_count: int,
) -> float:
    """
    Approximate the fraction of new geometry introduced by a generative
    pass.  Real implementation would diff vertex sets — this is good
    enough for the policy gate (clamped to [0,1]).
    """
    if completed_face_count <= 0:
        return 0.0
    if original_face_count <= 0:
        return 1.0
    delta = max(0, completed_face_count - original_face_count)
    return float(min(1.0, delta / max(completed_face_count, 1)))
