"""Sprint 3 — adaptive_sampling tests."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from modules.capture_workflow.adaptive_sampling import (
    AdaptiveSampler,
    SamplingThresholds,
    SamplingVerdict,
    _bbox_iou,
    _laplacian_variance,
    _optical_flow_median_magnitude,
)


def _sharp(seed=0, h=128, w=128):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _shifted(base: np.ndarray, dx: int) -> np.ndarray:
    out = np.zeros_like(base)
    h, w = base.shape[:2]
    if dx >= 0:
        out[:, dx:] = base[:, : w - dx]
    else:
        out[:, : w + dx] = base[:, -dx:]
    return out


def test_bbox_iou_basic():
    a = {"x": 0, "y": 0, "w": 10, "h": 10}
    b = {"x": 5, "y": 0, "w": 10, "h": 10}
    iou = _bbox_iou(a, b)
    assert 0.30 < iou < 0.40
    # Identical → 1.0
    assert _bbox_iou(a, a) == 1.0
    # Disjoint → 0.0
    assert _bbox_iou(a, {"x": 100, "y": 100, "w": 10, "h": 10}) == 0.0
    # None → 0.0
    assert _bbox_iou(None, a) == 0.0


def test_optical_flow_static_returns_zero():
    img = _sharp()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mag = _optical_flow_median_magnitude(gray, gray.copy())
    # Some noise from corner detection on identical images
    assert mag < 0.5


def test_optical_flow_shifted_picks_up_motion():
    img = _sharp(seed=42)
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shifted = _shifted(img, dx=8)
    gray2 = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    mag = _optical_flow_median_magnitude(gray1, gray2)
    assert mag > 5.0


def test_first_frame_always_kept():
    sampler = AdaptiveSampler()
    d = sampler.decide(_sharp())
    assert d.verdict == SamplingVerdict.KEEP_FORCED
    assert "first frame" in d.reasons[0]


def test_static_repeat_skipped():
    sampler = AdaptiveSampler()
    img = _sharp(seed=1)
    sampler.decide(img)  # first → keep
    d2 = sampler.decide(img.copy())
    assert d2.verdict == SamplingVerdict.SKIP_STATIC
    assert d2.flow_median < 1.5


def test_motion_burst_tagged():
    sampler = AdaptiveSampler(thresholds=SamplingThresholds(burst_flow=10.0))
    img = _sharp(seed=2)
    sampler.decide(img)
    d2 = sampler.decide(_shifted(img, dx=20))
    # 20px shift triggers burst tag (flow > 10)
    assert d2.verdict in (SamplingVerdict.KEEP_MOTION_BURST, SamplingVerdict.KEEP)


def test_blurry_skipped():
    sampler = AdaptiveSampler()
    sharp = _sharp(seed=3)
    sampler.decide(sharp)  # keep
    blurred = cv2.GaussianBlur(sharp, (21, 21), 8.0)
    d = sampler.decide(blurred)
    # Blurred image → low Laplacian variance → SKIP_BLURRY (or static if motion=0)
    assert d.verdict in (SamplingVerdict.SKIP_BLURRY, SamplingVerdict.SKIP_STATIC)


def test_force_keep_cadence():
    th = SamplingThresholds(force_keep_every_n_raw_frames=3)
    sampler = AdaptiveSampler(thresholds=th)
    img = _sharp(seed=5)
    sampler.decide(img)  # 0 → forced keep
    sampler.decide(img.copy())  # 1 → static skip
    sampler.decide(img.copy())  # 2 → static skip
    d = sampler.decide(img.copy())  # 3 → force-keep cadence kicks in
    assert d.verdict == SamplingVerdict.KEEP_FORCED
    assert any("force_keep_every_n" in r for r in d.reasons)


def test_stats_recorded():
    sampler = AdaptiveSampler()
    img = _sharp(seed=7)
    for _ in range(5):
        sampler.decide(img)
    s = sampler.stats.to_dict()
    assert s["kept_count"] >= 1
    assert sum(s["decisions"].values()) >= 5


def test_reset_clears_state():
    sampler = AdaptiveSampler()
    sampler.decide(_sharp(seed=0))
    assert sampler.last_kept_gray is not None
    sampler.reset()
    assert sampler.last_kept_gray is None
    assert sampler.stats.kept_count == 0
