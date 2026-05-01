"""
Azimuth diversity estimator — heuristic per-frame yaw bucketing.

Same constraint as elevation_estimator: at extraction time we don't have
COLMAP poses.  We infer relative yaw from the mask centroid horizontal
shift over time (proxy for camera orbit progress) and crude scene-content
hash diversity.

Caveats:
    - No absolute angle.  We only assert "frames spread over enough yaw
      to cover an orbit" not "frames at azimuth=42°".
    - Works best for orbit captures.  Linear pans look like one bucket.

After Sprint 4's intrinsics_cache lands we can lift this out to use
real BA-output yaws.  For now the heuristic is good enough to refuse
captures that are essentially one fixed viewpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import json

import numpy as np


_AZIMUTH_BUCKETS = 8


@dataclass
class AzimuthReport:
    frame_count: int
    centroid_x_norm: List[float] = field(default_factory=list)
    cumulative_orbit_progress: float = 0.0   # 0..1; 1.0 ≈ full orbit
    estimated_buckets_filled: int = 0
    estimated_coverage_ratio: float = 0.0
    max_consecutive_static_frames: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_centroid_x_norm(masks_dir: Path, frame_name: str) -> Optional[float]:
    stem = Path(frame_name).stem
    cand = masks_dir / f"{stem}.json"
    if not cand.exists():
        return None
    try:
        with open(cand, "r", encoding="utf-8") as f:
            meta = json.load(f)
        c = meta.get("centroid")
        if isinstance(c, dict):
            return c.get("x")
    except Exception:
        pass
    return None


def estimate_azimuth_distribution(
    frame_paths: List[str],
    masks_dir: Optional[Path] = None,
    static_motion_threshold: float = 0.01,
) -> AzimuthReport:
    """
    Use the centroid's normalized x-position trajectory as a proxy for
    orbit progress: net displacement + sign-change count map to bucket fill.
    """
    rep = AzimuthReport(frame_count=len(frame_paths))
    if not frame_paths:
        rep.notes.append("no frames")
        return rep

    if masks_dir is None:
        rep.notes.append("no masks_dir — azimuth diversity unmeasurable")
        return rep

    xs: List[float] = []
    for fp in frame_paths:
        cx = _load_centroid_x_norm(masks_dir, Path(fp).name)
        if cx is None:
            xs.append(0.5)  # neutral
            continue
        # Normalize: meta centroid is in pixels — but without frame width we cannot normalize reliably.
        # Fallback: assume already in [0, image_width]; normalize by max observed.
        xs.append(float(cx))

    if not xs:
        rep.notes.append("no centroid data")
        return rep

    arr = np.asarray(xs, dtype=float)
    if arr.max() > 1.5:  # heuristic: pixel coords, not normalized
        arr = arr / max(arr.max(), 1.0)
    rep.centroid_x_norm = arr.tolist()

    # Cumulative absolute motion (bigger = more orbit progress)
    diffs = np.abs(np.diff(arr))
    cumulative_motion = float(np.sum(diffs))
    rep.cumulative_orbit_progress = float(min(1.0, cumulative_motion))

    # Count zero-crossings around mean → proxy for rotation completing arcs
    centered = arr - float(np.mean(arr))
    sign_changes = int(np.sum(np.diff(np.sign(centered)) != 0))
    rep.estimated_buckets_filled = int(min(_AZIMUTH_BUCKETS, max(1, sign_changes + 1)))
    rep.estimated_coverage_ratio = rep.estimated_buckets_filled / _AZIMUTH_BUCKETS

    # Longest run of "static" frames (centroid barely moves)
    static_run = 0
    longest_static = 0
    for d in diffs:
        if d < static_motion_threshold:
            static_run += 1
            longest_static = max(longest_static, static_run)
        else:
            static_run = 0
    rep.max_consecutive_static_frames = int(longest_static)

    if rep.cumulative_orbit_progress < 0.30:
        rep.notes.append("low cumulative motion — capture may be a fixed-viewpoint pan")
    if rep.max_consecutive_static_frames > rep.frame_count * 0.3:
        rep.notes.append(
            f"{rep.max_consecutive_static_frames} consecutive nearly-static frames "
            f"(>{int(rep.frame_count*0.3)}) — operator paused mid-orbit"
        )

    return rep
