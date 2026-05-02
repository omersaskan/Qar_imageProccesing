"""
Pose geometry helpers — quaternion / translation → camera centre in world space.

All math is pure Python / stdlib; no numpy required so the module loads even
in minimal environments.  When numpy IS available it is used for the matrix
inversion (faster + numerically cleaner).

COLMAP convention:
  camera centre = -R^T @ t
  where R is the rotation matrix derived from qvec = [qw, qx, qy, qz].
"""
from __future__ import annotations

import math
from typing import List, Tuple


def qvec_to_rotation_matrix(qvec: List[float]) -> List[List[float]]:
    """
    Convert COLMAP quaternion [qw, qx, qy, qz] to 3×3 rotation matrix.
    Returns row-major nested list.
    """
    qw, qx, qy, qz = qvec
    # Normalise
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n < 1e-12:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n

    R = [
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),       2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),       1 - 2*(qx*qx + qy*qy)],
    ]
    return R


def camera_center_from_pose(qvec: List[float], tvec: List[float]) -> Tuple[float, float, float]:
    """
    Compute world-space camera centre from COLMAP pose (qvec, tvec).

    centre = -R^T @ t
    """
    R = qvec_to_rotation_matrix(qvec)
    # R^T @ (-t)
    tx, ty, tz = tvec
    cx = -(R[0][0]*tx + R[1][0]*ty + R[2][0]*tz)
    cy = -(R[0][1]*tx + R[1][1]*ty + R[2][1]*tz)
    cz = -(R[0][2]*tx + R[1][2]*ty + R[2][2]*tz)
    return cx, cy, cz


def cartesian_to_spherical(cx: float, cy: float, cz: float) -> Tuple[float, float, float]:
    """
    Convert Cartesian camera centre to spherical coordinates relative to origin.

    Returns (radius, azimuth_deg, elevation_deg).
    Azimuth: 0° = +X axis, increases counter-clockwise in XY plane.
    Elevation: 0° = XY plane, +90° = +Z (above subject).
    """
    r = math.sqrt(cx*cx + cy*cy + cz*cz)
    if r < 1e-12:
        return 0.0, 0.0, 0.0
    azimuth_rad = math.atan2(cy, cx)
    elevation_rad = math.asin(max(-1.0, min(1.0, cz / r)))
    return r, math.degrees(azimuth_rad), math.degrees(elevation_rad)


def compute_scene_centroid(centers: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Mean position of all camera centres."""
    if not centers:
        return 0.0, 0.0, 0.0
    n = len(centers)
    mx = sum(c[0] for c in centers) / n
    my = sum(c[1] for c in centers) / n
    mz = sum(c[2] for c in centers) / n
    return mx, my, mz


def centres_relative_to_centroid(
    centers: List[Tuple[float, float, float]],
    centroid: Tuple[float, float, float],
) -> List[Tuple[float, float, float]]:
    """Translate camera centres to scene-centroid frame."""
    ox, oy, oz = centroid
    return [(cx - ox, cy - oy, cz - oz) for cx, cy, cz in centers]
