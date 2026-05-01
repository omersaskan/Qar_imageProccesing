"""
Blur burst detector — find and prune temporally-clustered blurry frames.

Single-frame blur is normal (operator pause, focus shift). Trouble is
*runs* of consecutive blurry frames — usually motion bursts during a
fast pan or hand jitter spike. They poison reconstruction: COLMAP gets
them anyway, BA drift accumulates.

Strategy:
    1. Compute Laplacian variance per frame (cheap, ~1ms / frame).
    2. Robust z-score against the per-session median absolute deviation
       (MAD), not the mean — outliers don't poison the threshold.
    3. Detect runs of N+ consecutive frames below z-threshold.
    4. Report bursts as (start_idx, end_idx, count) — caller decides to
       drop, blank-mask, or warn.

Used after frame extraction; orthogonal to per-frame quality_analyzer
(which already filters single bad frames).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class BlurBurst:
    start_index: int
    end_index: int
    count: int
    median_score: float
    frames: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BlurBurstReport:
    frame_count: int
    scores: List[float] = field(default_factory=list)
    median_score: float = 0.0
    mad: float = 0.0
    z_threshold: float = -1.5
    min_burst_length: int = 3
    bursts: List[BlurBurst] = field(default_factory=list)
    total_burst_frames: int = 0
    burst_ratio: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "scores": self.scores,
            "median_score": self.median_score,
            "mad": self.mad,
            "z_threshold": self.z_threshold,
            "min_burst_length": self.min_burst_length,
            "bursts": [b.to_dict() for b in self.bursts],
            "total_burst_frames": self.total_burst_frames,
            "burst_ratio": self.burst_ratio,
            "notes": self.notes,
        }


def _read_gray(path: Path) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_blur_scores(frame_paths: List[str], downsample: int = 2) -> List[float]:
    """
    Per-frame Laplacian variance.  Higher = sharper.
    Downsample (default 2) speeds up by 4x with ~no quality loss for blur metric.
    Returns 0.0 for unreadable frames so indices stay aligned.
    """
    out: List[float] = []
    for fp in frame_paths:
        gray = _read_gray(Path(fp))
        if gray is None:
            out.append(0.0)
            continue
        if downsample > 1:
            gray = gray[::downsample, ::downsample]
        out.append(_laplacian_variance(gray))
    return out


def _robust_z_scores(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Modified z-score using MAD; robust to outliers (Iglewicz & Hoaglin)."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad < 1e-9:
        # Degenerate: all values nearly identical
        return np.zeros_like(values, dtype=float), median, mad
    z = 0.6745 * (values - median) / mad
    return z, median, mad


def detect_bursts(
    frame_paths: List[str],
    z_threshold: float = -1.5,
    min_burst_length: int = 3,
    downsample: int = 2,
) -> BlurBurstReport:
    """
    Compute scores, then return runs of `min_burst_length`+ consecutive
    frames whose modified z-score is below `z_threshold` (i.e. unusually blurry).

    z_threshold defaults:
        -1.5 → reasonably aggressive (outliers + nearly-outliers caught)
        -2.5 → conservative (only egregious bursts)
    """
    scores = compute_blur_scores(frame_paths, downsample=downsample)
    rep = BlurBurstReport(
        frame_count=len(frame_paths),
        scores=[float(s) for s in scores],
        z_threshold=z_threshold,
        min_burst_length=min_burst_length,
    )
    if rep.frame_count == 0:
        rep.notes.append("no frames")
        return rep

    arr = np.asarray(scores, dtype=float)
    z, median, mad = _robust_z_scores(arr)
    rep.median_score = median
    rep.mad = mad

    # Identify "blurry" frames (negative z below threshold)
    blurry_flags = z < z_threshold

    # Find runs
    bursts: List[BlurBurst] = []
    i = 0
    n = len(blurry_flags)
    while i < n:
        if not blurry_flags[i]:
            i += 1
            continue
        start = i
        while i < n and blurry_flags[i]:
            i += 1
        run_len = i - start
        if run_len >= min_burst_length:
            burst_scores = arr[start:i]
            bursts.append(BlurBurst(
                start_index=int(start),
                end_index=int(i - 1),
                count=int(run_len),
                median_score=float(np.median(burst_scores)),
                frames=[frame_paths[k] for k in range(start, i)],
            ))

    rep.bursts = bursts
    rep.total_burst_frames = sum(b.count for b in bursts)
    rep.burst_ratio = rep.total_burst_frames / max(rep.frame_count, 1)

    if mad < 1e-9:
        rep.notes.append("MAD≈0 — no spread, bursts cannot be detected reliably")
    if rep.burst_ratio > 0.20:
        rep.notes.append(f"burst_ratio {rep.burst_ratio:.0%} >20% — wholesale capture quality issue")

    return rep
