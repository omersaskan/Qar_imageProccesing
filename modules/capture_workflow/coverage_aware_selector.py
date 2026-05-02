"""
Coverage-aware keyframe selector — second pass after adaptive sampling.

Adaptive sampler picks "interesting" frames; this layer **rebalances** the
selection so under-represented elevation/azimuth buckets get prioritized.

Use case: the operator captures 50 sharp orbit frames at eye level (mid
band) plus 5 brief top-down frames.  AdaptiveSampler keeps maybe 20
mid + 3 top.  Reconstruction then has poor top coverage → mesh holes.

This selector caps over-represented buckets and forcibly retains under-
represented ones, even if their adaptive_sampling sharpness is borderline.

Inputs:
    - candidate frames (post-adaptive)
    - per-frame bucket assignments (from elevation_estimator + azimuth_diversity)
    - target counts per bucket (from CaptureProfile or defaults)

Output:
    - rebalanced frame list
    - rebalance report (kept counts per bucket, dropped reasons)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CoverageTargets:
    # Min frames per elevation bucket
    min_low: int = 6
    min_mid: int = 10
    min_top: int = 6
    # Max total kept frames (texture pipeline budget)
    max_total: int = 80
    # Per-bucket cap to prevent one band dominating
    per_bucket_cap_ratio: float = 0.55


@dataclass
class FrameAssignment:
    frame_path: str
    elevation_bucket: str   # low | mid | top | unknown
    sharpness: float = 0.0
    confidence: float = 0.0


@dataclass
class SelectionReport:
    candidate_count: int
    kept_count: int
    bucket_counts_before: Dict[str, int] = field(default_factory=dict)
    bucket_counts_after: Dict[str, int] = field(default_factory=dict)
    targets: Dict[str, Any] = field(default_factory=dict)
    rebalance_actions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def select_balanced_frames(
    assignments: List[FrameAssignment],
    targets: Optional[CoverageTargets] = None,
) -> Tuple[List[str], SelectionReport]:
    """
    Trim or reorder `assignments` to satisfy bucket targets.

    Strategy:
        1. Bucket the candidates.
        2. Within each bucket, sort by sharpness * confidence (best first).
        3. Cap each bucket at `per_bucket_cap_ratio * max_total`.
        4. Take top-K from each bucket up to its min target.
        5. Fill remaining slots with leftover sorted-quality picks.
    """
    t = targets or CoverageTargets()
    rep = SelectionReport(candidate_count=len(assignments), kept_count=0,
                          targets=asdict(t))

    if not assignments:
        rep.notes.append("no candidates")
        return [], rep

    # Bucket
    bucketed: Dict[str, List[FrameAssignment]] = {"low": [], "mid": [], "top": [], "unknown": []}
    for a in assignments:
        bucketed.setdefault(a.elevation_bucket, []).append(a)
    rep.bucket_counts_before = {k: len(v) for k, v in bucketed.items()}

    # Sort each bucket by quality
    for k in bucketed:
        bucketed[k].sort(key=lambda x: x.sharpness * max(x.confidence, 0.1), reverse=True)

    cap_per_bucket = max(1, int(t.max_total * t.per_bucket_cap_ratio))
    minima = {"low": t.min_low, "mid": t.min_mid, "top": t.min_top, "unknown": 0}

    kept: List[FrameAssignment] = []

    # Phase 1: take min targets from each bucket
    for k in ("low", "mid", "top"):
        avail = bucketed[k]
        target = min(minima[k], len(avail))
        if target < minima[k]:
            rep.rebalance_actions.append(
                f"bucket '{k}' under-represented: only {len(avail)} candidates < min {minima[k]}"
            )
        if target > 0:
            kept.extend(avail[:target])
            bucketed[k] = avail[target:]
            rep.rebalance_actions.append(f"reserved {target} from '{k}'")

    # Phase 2: fill remaining slots round-robin by quality, respecting caps
    remaining_slots = max(0, t.max_total - len(kept))
    flat = []
    for k in ("low", "mid", "top", "unknown"):
        for a in bucketed[k]:
            flat.append((a.sharpness * max(a.confidence, 0.1), a))
    flat.sort(key=lambda x: x[0], reverse=True)

    bucket_counts = {k: sum(1 for x in kept if x.elevation_bucket == k) for k in ("low", "mid", "top", "unknown")}
    for _, a in flat:
        if remaining_slots <= 0:
            break
        if bucket_counts.get(a.elevation_bucket, 0) >= cap_per_bucket and a.elevation_bucket != "unknown":
            continue
        kept.append(a)
        bucket_counts[a.elevation_bucket] = bucket_counts.get(a.elevation_bucket, 0) + 1
        remaining_slots -= 1

    rep.kept_count = len(kept)
    rep.bucket_counts_after = bucket_counts

    if rep.kept_count == 0:
        rep.notes.append("rebalancer produced no frames — fallback expected")

    return [a.frame_path for a in kept], rep
