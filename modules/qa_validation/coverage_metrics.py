"""
Coverage metrics for the QA scorecard.

Numbers we care about per reconstruction job:
    - observed_surface_ratio    : how much of the mesh is backed by capture
    - azimuth_coverage_ratio    : 8 yaw buckets — % filled
    - elevation_coverage_ratio  : 5 pitch buckets — % filled
    - multi_height_score        : low/mid/top buckets — discrete 0/0.33/0.67/1.0
    - max_azimuth_gap_deg       : largest empty arc (frame-rate gap detector)
    - view_diversity_score      : weighted combo, 0..1

Inputs are deliberately minimal: list of camera objects (or EXIF-loaded poses)
and an optional mesh.  The module degrades gracefully — partial inputs return
whatever metrics they can compute.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import math

import numpy as np


# Azimuth: 8 buckets of 45° each, full circle
_AZIMUTH_BUCKETS = 8
# Elevation: -30° to +90°, 5 buckets
_ELEVATION_RANGE = (-30.0, 90.0)
_ELEVATION_BUCKETS = 5

# Multi-height buckets: low (≤30°), mid (30-60°), top (>60°)
_MULTI_HEIGHT_LOW_MAX = 30.0
_MULTI_HEIGHT_MID_MAX = 60.0


@dataclass
class CoverageReport:
    sample_count: int = 0
    observed_surface_ratio: float = 0.0
    observed_surface_method: str = "none"  # point_cloud | mask_projection | heuristic | none
    azimuth_coverage_ratio: float = 0.0
    azimuth_buckets_filled: int = 0
    azimuth_buckets_total: int = _AZIMUTH_BUCKETS
    max_azimuth_gap_deg: float = 360.0
    elevation_coverage_ratio: float = 0.0
    elevation_buckets_filled: int = 0
    elevation_buckets_total: int = _ELEVATION_BUCKETS
    multi_height_score: float = 0.0
    multi_height_buckets: Dict[str, int] = field(default_factory=lambda: {"low": 0, "mid": 0, "top": 0})
    view_diversity_score: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _camera_position(cam: Any) -> Optional[np.ndarray]:
    """Pull (x,y,z) world position from a camera dict or COLMAP-style record."""
    if isinstance(cam, dict):
        for key in ("position", "center", "C", "translation"):
            v = cam.get(key)
            if v is not None:
                arr = np.asarray(v, dtype=float).flatten()
                if arr.size == 3:
                    return arr
        # Fall back to qvec/tvec → -R^T * t
        qvec = cam.get("qvec") or cam.get("rotation_quat")
        tvec = cam.get("tvec") or cam.get("translation_vec")
        if qvec is not None and tvec is not None:
            try:
                from scipy.spatial.transform import Rotation as R
                rot = R.from_quat(np.asarray(qvec)[[1, 2, 3, 0]]).as_matrix()
                t = np.asarray(tvec, dtype=float).reshape(3)
                return -rot.T @ t
            except Exception:
                return None
    elif hasattr(cam, "position"):
        return np.asarray(cam.position, dtype=float)
    return None


def _yaw_pitch_from_position(pos: np.ndarray, center: np.ndarray) -> Optional[tuple]:
    """Spherical coords around `center`. Returns (yaw_deg, pitch_deg)."""
    delta = pos - center
    r_h = math.hypot(delta[0], delta[1])
    if r_h < 1e-9 and abs(delta[2]) < 1e-9:
        return None
    yaw = math.degrees(math.atan2(delta[1], delta[0])) % 360.0
    pitch = math.degrees(math.atan2(delta[2], r_h))
    return yaw, pitch


def _azimuth_metrics(yaws: List[float]) -> Dict[str, Any]:
    if not yaws:
        return {
            "azimuth_coverage_ratio": 0.0,
            "azimuth_buckets_filled": 0,
            "max_azimuth_gap_deg": 360.0,
        }
    bucket_size = 360.0 / _AZIMUTH_BUCKETS
    buckets = set()
    for y in yaws:
        buckets.add(int((y % 360.0) // bucket_size))
    filled = len(buckets)

    # Compute the largest empty arc from sorted yaws
    ys = sorted(y % 360.0 for y in yaws)
    gaps = [(ys[(i + 1) % len(ys)] - ys[i]) % 360.0 for i in range(len(ys))]
    if len(ys) > 1:
        max_gap = max(gaps)
    else:
        max_gap = 360.0
    return {
        "azimuth_coverage_ratio": filled / _AZIMUTH_BUCKETS,
        "azimuth_buckets_filled": filled,
        "max_azimuth_gap_deg": float(max_gap),
    }


def _elevation_metrics(pitches: List[float]) -> Dict[str, Any]:
    if not pitches:
        return {
            "elevation_coverage_ratio": 0.0,
            "elevation_buckets_filled": 0,
            "multi_height_score": 0.0,
            "multi_height_buckets": {"low": 0, "mid": 0, "top": 0},
        }
    lo, hi = _ELEVATION_RANGE
    bucket_size = (hi - lo) / _ELEVATION_BUCKETS
    buckets = set()
    counts = {"low": 0, "mid": 0, "top": 0}
    for p in pitches:
        if p < lo:
            p_clamped = lo
        elif p > hi:
            p_clamped = hi
        else:
            p_clamped = p
        idx = min(_ELEVATION_BUCKETS - 1, int((p_clamped - lo) // bucket_size))
        buckets.add(idx)
        if p < _MULTI_HEIGHT_LOW_MAX:
            counts["low"] += 1
        elif p < _MULTI_HEIGHT_MID_MAX:
            counts["mid"] += 1
        else:
            counts["top"] += 1
    filled = len(buckets)
    # multi-height: each non-empty bucket worth 1/3 (capped at 1.0)
    mh = sum(1 for v in counts.values() if v > 0) / 3.0
    return {
        "elevation_coverage_ratio": filled / _ELEVATION_BUCKETS,
        "elevation_buckets_filled": filled,
        "multi_height_score": float(mh),
        "multi_height_buckets": counts,
    }


def compute_coverage_report(
    cameras: Optional[List[Any]] = None,
    mesh: Optional[Any] = None,
    masks: Optional[Dict[str, Any]] = None,
    point_cloud: Optional[Any] = None,
) -> CoverageReport:
    """
    Build a CoverageReport from whatever inputs are available.
    All inputs optional; missing pieces simply omit those fields.
    """
    rep = CoverageReport()
    notes: List[str] = []

    # 1. Observed surface — delegate to ai_completion.coverage helper
    if mesh is not None:
        try:
            from modules.ai_completion.coverage import compute_observed_surface_ratio
            cov = compute_observed_surface_ratio(
                mesh, cameras=cameras, masks=masks, point_cloud=point_cloud
            )
            rep.observed_surface_ratio = float(cov.get("observed_surface_ratio", 0.0))
            rep.observed_surface_method = cov.get("method", "none")
        except Exception as e:
            notes.append(f"observed_surface failed: {e}")

    # 2. Azimuth + elevation buckets from camera poses
    yaws: List[float] = []
    pitches: List[float] = []
    positions: List[np.ndarray] = []
    if cameras:
        for cam in cameras:
            pos = _camera_position(cam)
            if pos is not None:
                positions.append(pos)
        if positions:
            center = np.mean(positions, axis=0)
            for pos in positions:
                yp = _yaw_pitch_from_position(pos, center)
                if yp:
                    yaws.append(yp[0])
                    pitches.append(yp[1])
        else:
            notes.append("no usable camera positions")

    az = _azimuth_metrics(yaws)
    el = _elevation_metrics(pitches)
    rep.sample_count = len(positions)
    rep.azimuth_coverage_ratio = az["azimuth_coverage_ratio"]
    rep.azimuth_buckets_filled = az["azimuth_buckets_filled"]
    rep.max_azimuth_gap_deg = az["max_azimuth_gap_deg"]
    rep.elevation_coverage_ratio = el["elevation_coverage_ratio"]
    rep.elevation_buckets_filled = el["elevation_buckets_filled"]
    rep.multi_height_score = el["multi_height_score"]
    rep.multi_height_buckets = el["multi_height_buckets"]

    # 3. Composite diversity (0..1) — weighted geo-mean style
    az_w = rep.azimuth_coverage_ratio
    el_w = rep.elevation_coverage_ratio
    obs_w = rep.observed_surface_ratio
    components = [v for v in (az_w, el_w, obs_w) if v > 0]
    if components:
        rep.view_diversity_score = float(sum(components) / 3.0)
    rep.notes = notes
    return rep
