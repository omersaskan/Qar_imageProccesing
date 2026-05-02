"""
Pose-backed 3×8 coverage matrix — Sprint 5.

Bins registered camera centres (from COLMAP sparse) into a 3-row × 8-column
grid where rows = elevation bands (low/mid/high) and columns = 8 azimuth
sectors of 45° each.

A cell is "covered" when at least one camera lands in that bin.

Returns a dict that can be written directly into the manifest under
`pose_backed_coverage`.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .colmap_sparse_parser import load_sparse_model
from .pose_geometry import (
    camera_center_from_pose,
    cartesian_to_spherical,
    compute_scene_centroid,
    centres_relative_to_centroid,
)

# Elevation band definitions (degrees).
ELEVATION_BANDS = [
    ("low",  -90.0, -15.0),
    ("mid",  -15.0,  30.0),
    ("high",  30.0,  90.0),
]
N_AZIMUTH_SECTORS = 8
AZIMUTH_SECTOR_DEG = 360.0 / N_AZIMUTH_SECTORS  # 45°


def _azimuth_sector(az_deg: float) -> int:
    """Map azimuth in [-180,180] to sector index [0, N_AZIMUTH_SECTORS)."""
    az_norm = az_deg % 360.0  # [0, 360)
    return int(az_norm / AZIMUTH_SECTOR_DEG) % N_AZIMUTH_SECTORS


def _elevation_band(el_deg: float) -> Optional[int]:
    """Return row index (0=low, 1=mid, 2=high) or None if out of all bands."""
    for i, (_, lo, hi) in enumerate(ELEVATION_BANDS):
        if lo <= el_deg < hi:
            return i
    # Clamp extremes to nearest band
    if el_deg < ELEVATION_BANDS[0][1]:
        return 0
    return len(ELEVATION_BANDS) - 1


def build_coverage_matrix(
    images: List[Dict],
    *,
    min_cameras: int = 1,
) -> Dict[str, Any]:
    """
    Build a 3×8 coverage matrix from a list of COLMAP image dicts
    (as returned by colmap_sparse_parser.parse_images_txt).

    Returns a result dict ready for manifest embedding.
    """
    if not images:
        return _unavailable("no registered images")

    # 1. Compute camera centres
    centers: List[Tuple[float, float, float]] = []
    for img in images:
        try:
            c = camera_center_from_pose(img["qvec"], img["tvec"])
            centers.append(c)
        except Exception:
            continue

    if len(centers) < min_cameras:
        return _unavailable(f"too few valid camera centres: {len(centers)}")

    # 2. Shift to scene centroid frame
    centroid = compute_scene_centroid(centers)
    rel_centers = centres_relative_to_centroid(centers, centroid)

    # 3. Fill matrix[row][col] = count
    matrix: List[List[int]] = [[0] * N_AZIMUTH_SECTORS for _ in range(len(ELEVATION_BANDS))]
    azimuths: List[float] = []
    elevations: List[float] = []

    for cx, cy, cz in rel_centers:
        r, az, el = cartesian_to_spherical(cx, cy, cz)
        if r < 1e-9:
            continue
        azimuths.append(az)
        elevations.append(el)
        row = _elevation_band(el)
        col = _azimuth_sector(az)
        if row is not None:
            matrix[row][col] += 1

    # 4. Derive metrics
    total_cells = len(ELEVATION_BANDS) * N_AZIMUTH_SECTORS
    covered_cells = sum(1 for r in matrix for v in r if v > 0)
    coverage_ratio = covered_cells / total_cells

    azimuth_span = _azimuth_span(azimuths)
    elevation_spread = (max(elevations) - min(elevations)) if len(elevations) >= 2 else 0.0

    band_coverage = {}
    for i, (band_name, _, _) in enumerate(ELEVATION_BANDS):
        covered = sum(1 for v in matrix[i] if v > 0)
        band_coverage[band_name] = {
            "covered_sectors": covered,
            "total_sectors": N_AZIMUTH_SECTORS,
            "sector_counts": matrix[i],
        }

    return {
        "status": "ok",
        "registered_count": len(images),
        "valid_centres": len(centers),
        "coverage_ratio": round(coverage_ratio, 4),
        "covered_cells": covered_cells,
        "total_cells": total_cells,
        "azimuth_span_degrees": round(azimuth_span, 2),
        "elevation_spread_degrees": round(elevation_spread, 2),
        "centroid": {"x": round(centroid[0], 6), "y": round(centroid[1], 6), "z": round(centroid[2], 6)},
        "bands": band_coverage,
        "matrix": matrix,
    }


def _azimuth_span(azimuths: List[float]) -> float:
    """Compute the angular span covered by azimuth samples (degrees)."""
    if not azimuths:
        return 0.0
    # Convert to [0, 360)
    norm = [a % 360.0 for a in azimuths]
    norm.sort()
    # Largest gap between consecutive (circular)
    gaps = [norm[i+1] - norm[i] for i in range(len(norm)-1)]
    gaps.append(360.0 - norm[-1] + norm[0])
    max_gap = max(gaps)
    return round(360.0 - max_gap, 2)


def _unavailable(reason: str) -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "coverage_ratio": 0.0,
        "covered_cells": 0,
        "total_cells": len(ELEVATION_BANDS) * N_AZIMUTH_SECTORS,
        "azimuth_span_degrees": 0.0,
        "elevation_spread_degrees": 0.0,
        "bands": {},
        "matrix": [],
    }


def coverage_from_attempt_dir(attempt_dir) -> Dict[str, Any]:
    """
    Top-level helper: load sparse model from attempt_dir and build coverage.

    Safe to call even when sparse output is missing — returns unavailable dict.
    """
    from pathlib import Path
    try:
        cameras, images, model_dir = load_sparse_model(Path(attempt_dir))
        if not images:
            return _unavailable("sparse model missing or empty")
        return build_coverage_matrix(images)
    except Exception as exc:
        import logging
        logging.warning(f"pose_coverage_matrix: {exc}")
        return _unavailable(str(exc)[:200])
