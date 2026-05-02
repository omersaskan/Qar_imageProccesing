"""Sprint 3 — coverage_aware_selector tests."""
from __future__ import annotations

import pytest

from modules.capture_workflow.coverage_aware_selector import (
    CoverageTargets,
    FrameAssignment,
    select_balanced_frames,
)


def _make(name: str, bucket: str, sharp: float = 1.0, conf: float = 0.7):
    return FrameAssignment(frame_path=name, elevation_bucket=bucket, sharpness=sharp, confidence=conf)


def test_no_candidates_returns_empty():
    kept, rep = select_balanced_frames([])
    assert kept == []
    assert rep.candidate_count == 0
    assert "no candidates" in " ".join(rep.notes)


def test_single_band_keeps_what_it_can():
    # All mid, no low/top
    candidates = [_make(f"f_{i}.jpg", "mid", sharp=10 - i) for i in range(20)]
    kept, rep = select_balanced_frames(candidates, CoverageTargets(min_low=5, min_mid=10, min_top=5))
    assert rep.bucket_counts_after.get("mid", 0) > 0
    # Should flag under-representation for low + top
    under = [a for a in rep.rebalance_actions if "under-represented" in a]
    assert any("low" in u for u in under)
    assert any("top" in u for u in under)


def test_three_bands_kept_balanced():
    candidates = []
    for b in ("low", "mid", "top"):
        for i in range(30):
            candidates.append(_make(f"{b}_{i}.jpg", b, sharp=float(30 - i)))
    kept, rep = select_balanced_frames(candidates, CoverageTargets(min_low=6, min_mid=10, min_top=6, max_total=40))
    assert rep.kept_count == 40
    # Each band represented
    for k in ("low", "mid", "top"):
        assert rep.bucket_counts_after[k] >= 6


def test_per_bucket_cap_respected():
    # Heavy mid bias, small low/top
    candidates = [_make(f"mid_{i}.jpg", "mid", sharp=float(100 - i)) for i in range(80)]
    candidates += [_make(f"low_{i}.jpg", "low", sharp=10) for i in range(8)]
    candidates += [_make(f"top_{i}.jpg", "top", sharp=10) for i in range(8)]
    targets = CoverageTargets(max_total=40, per_bucket_cap_ratio=0.50)  # cap=20 per bucket
    kept, rep = select_balanced_frames(candidates, targets)
    assert rep.bucket_counts_after["mid"] <= 20  # cap enforced


def test_quality_ordering_within_bucket():
    candidates = [
        _make("low_dim.jpg", "low", sharp=1.0),
        _make("low_bright.jpg", "low", sharp=10.0),
        _make("mid_dim.jpg", "mid", sharp=2.0),
        _make("mid_bright.jpg", "mid", sharp=20.0),
    ]
    kept, rep = select_balanced_frames(
        candidates, CoverageTargets(min_low=1, min_mid=1, min_top=0, max_total=2)
    )
    # Top-1 from low should be the brightest
    assert "low_bright.jpg" in kept
    assert "mid_bright.jpg" in kept
