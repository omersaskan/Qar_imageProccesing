"""
Reconstruction Preflight — last-mile sanity gate before COLMAP starts.

Checks hand-off quality from Sprint 1-3 outputs (selected keyframes +
capture_gate + frame stats) and decides whether to:

    - pass    : run reconstruction with the resolver-picked preset
    - review  : run reconstruction with `baseline` preset (be safe)
    - reject  : do not run; return capture_quality_rejected

Reject is deliberately conservative — only frame-level catastrophes
(no readable frames, dimensions inconsistent, frame count below 3).
Most quality issues bubble up as "review".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class PreflightDecision(str, Enum):
    PASS = "pass"
    REVIEW = "review"
    REJECT = "reject"


@dataclass
class PreflightThresholds:
    min_frames_hard: int = 3                    # below this → reject (COLMAP can't bootstrap)
    min_frames_review: int = 8                  # below this → review (results unreliable)
    min_coverage_ratio_review: float = 0.30
    min_coverage_ratio_reject: float = 0.10
    min_blur_median_review: float = 30.0        # Laplacian variance
    min_blur_median_reject: float = 10.0
    max_static_run_ratio_review: float = 0.50
    max_static_run_ratio_reject: float = 0.85
    max_dimension_mismatch_ratio: float = 0.20  # >20% frames with different shape → reject


@dataclass
class PreflightReport:
    decision: PreflightDecision = PreflightDecision.PASS
    reasons: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    selected_count: int = 0
    coverage_ratio: float = 0.0
    median_blur_score: float = 0.0
    static_run_ratio: float = 0.0
    dimension_mismatch_ratio: float = 0.0
    missing_frame_count: int = 0
    thresholds: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


def _quick_dimension_scan(
    frame_paths: List[str], max_samples: int = 10
) -> Tuple[int, int, float]:
    """
    Returns (missing_count, mismatch_count, mismatch_ratio).
    Reads up to `max_samples` frames; uses cv2 imdecode for unicode-safe paths.
    """
    if not frame_paths:
        return 0, 0, 0.0
    total = len(frame_paths)
    if total > max_samples:
        step = total / max_samples
        sample = [frame_paths[int(i * step)] for i in range(max_samples)]
    else:
        sample = list(frame_paths)

    missing = 0
    shapes: List[Tuple[int, int]] = []
    for p in sample:
        path = Path(p)
        if not path.exists():
            missing += 1
            continue
        try:
            buf = np.fromfile(str(path), dtype=np.uint8)
            if buf.size == 0:
                missing += 1
                continue
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if img is None:
                missing += 1
                continue
            shapes.append(img.shape[:2])
        except Exception:
            missing += 1
    if not shapes:
        return missing, 0, 0.0
    # Most common shape
    counts: Dict[Tuple[int, int], int] = {}
    for s in shapes:
        counts[s] = counts.get(s, 0) + 1
    majority = max(counts.values())
    mismatch = len(shapes) - majority
    mismatch_ratio = mismatch / max(len(shapes), 1)
    return missing, mismatch, mismatch_ratio


def _coverage_ratio_from_gate(capture_gate: Optional[Dict[str, Any]]) -> float:
    """3×8 matrix → fraction of cells filled."""
    if not capture_gate:
        return 0.0
    m = capture_gate.get("matrix_3x8") or []
    if not m:
        return 0.0
    total = 0
    filled = 0
    for row in m:
        for v in row:
            total += 1
            if v and v > 0:
                filled += 1
    return filled / max(total, 1)


def _median_blur_from_gate(capture_gate: Optional[Dict[str, Any]]) -> float:
    if not capture_gate:
        return 0.0
    blur = capture_gate.get("blur") or {}
    return float(blur.get("median_score", 0.0) or 0.0)


def _static_run_ratio_from_gate(capture_gate: Optional[Dict[str, Any]]) -> float:
    if not capture_gate:
        return 0.0
    az = capture_gate.get("azimuth") or {}
    n = int(az.get("frame_count", 0) or 0)
    s = int(az.get("max_consecutive_static_frames", 0) or 0)
    return s / max(n, 1)


def _worse(a: PreflightDecision, b: PreflightDecision) -> PreflightDecision:
    rank = {PreflightDecision.PASS: 0, PreflightDecision.REVIEW: 1, PreflightDecision.REJECT: 2}
    return b if rank[b] > rank[a] else a


def evaluate_preflight(
    selected_keyframes: List[str],
    capture_gate: Optional[Dict[str, Any]] = None,
    thresholds: Optional[PreflightThresholds] = None,
) -> PreflightReport:
    th = thresholds or PreflightThresholds()
    rep = PreflightReport(thresholds=asdict(th))

    rep.selected_count = len(selected_keyframes)
    rep.coverage_ratio = _coverage_ratio_from_gate(capture_gate)
    rep.median_blur_score = _median_blur_from_gate(capture_gate)
    rep.static_run_ratio = _static_run_ratio_from_gate(capture_gate)

    # 1. Hard reject — frame count too low
    if rep.selected_count == 0:
        rep.decision = PreflightDecision.REJECT
        rep.reasons.append("no selected keyframes")
        rep.suggestions.append("re-capture with sufficient duration / coverage")
        return rep

    if rep.selected_count < th.min_frames_hard:
        rep.decision = PreflightDecision.REJECT
        rep.reasons.append(f"selected_count {rep.selected_count} < hard min {th.min_frames_hard}")
        rep.suggestions.append("COLMAP needs at least 3 readable frames")
        return rep

    # 2. Frame-level checks (paths exist + dimensions match)
    missing, mismatch, mismatch_ratio = _quick_dimension_scan(selected_keyframes)
    rep.missing_frame_count = missing
    rep.dimension_mismatch_ratio = mismatch_ratio

    if missing >= max(1, rep.selected_count // 3):
        rep.decision = PreflightDecision.REJECT
        rep.reasons.append(f"{missing} sampled frames missing/unreadable (≥{rep.selected_count // 3})")
        rep.suggestions.append("frame extraction may have written zero-byte files")
        return rep

    if mismatch_ratio > th.max_dimension_mismatch_ratio:
        rep.decision = PreflightDecision.REJECT
        rep.reasons.append(f"dimension_mismatch_ratio {mismatch_ratio:.0%} > {th.max_dimension_mismatch_ratio:.0%}")
        rep.suggestions.append("re-encode the source video to a uniform resolution")
        return rep

    # 3. Soft signals — degrade to review
    if rep.selected_count < th.min_frames_review:
        rep.decision = _worse(rep.decision, PreflightDecision.REVIEW)
        rep.reasons.append(
            f"selected_count {rep.selected_count} < review threshold {th.min_frames_review}"
        )
        rep.suggestions.append("results may be unreliable — operator should review")

    if capture_gate:
        if rep.coverage_ratio > 0 and rep.coverage_ratio < th.min_coverage_ratio_reject:
            rep.decision = _worse(rep.decision, PreflightDecision.REJECT)
            rep.reasons.append(
                f"coverage_ratio {rep.coverage_ratio:.2f} < hard reject {th.min_coverage_ratio_reject:.2f}"
            )
            rep.suggestions.append("capture matrix mostly empty — reshoot with multi-height + orbit")
        elif rep.coverage_ratio > 0 and rep.coverage_ratio < th.min_coverage_ratio_review:
            rep.decision = _worse(rep.decision, PreflightDecision.REVIEW)
            rep.reasons.append(
                f"coverage_ratio {rep.coverage_ratio:.2f} below review threshold "
                f"{th.min_coverage_ratio_review:.2f}"
            )

        if rep.median_blur_score > 0 and rep.median_blur_score < th.min_blur_median_reject:
            rep.decision = _worse(rep.decision, PreflightDecision.REJECT)
            rep.reasons.append(
                f"median_blur_score {rep.median_blur_score:.1f} < hard reject {th.min_blur_median_reject}"
            )
            rep.suggestions.append("frames are too blurry for SIFT — better lighting or steadier capture")
        elif rep.median_blur_score > 0 and rep.median_blur_score < th.min_blur_median_review:
            rep.decision = _worse(rep.decision, PreflightDecision.REVIEW)
            rep.reasons.append(
                f"median_blur_score {rep.median_blur_score:.1f} below review threshold "
                f"{th.min_blur_median_review}"
            )

        if rep.static_run_ratio > th.max_static_run_ratio_reject:
            rep.decision = _worse(rep.decision, PreflightDecision.REJECT)
            rep.reasons.append(
                f"static_run_ratio {rep.static_run_ratio:.0%} > hard reject "
                f"{th.max_static_run_ratio_reject:.0%}"
            )
            rep.suggestions.append("scene was effectively stationary — orbit the object")
        elif rep.static_run_ratio > th.max_static_run_ratio_review:
            rep.decision = _worse(rep.decision, PreflightDecision.REVIEW)
            rep.reasons.append(
                f"static_run_ratio {rep.static_run_ratio:.0%} above review threshold "
                f"{th.max_static_run_ratio_review:.0%}"
            )

    if not rep.reasons:
        rep.notes.append("all preflight checks within thresholds")

    return rep
