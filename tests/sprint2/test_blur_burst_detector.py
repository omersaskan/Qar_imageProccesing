"""Sprint 2 — blur_burst_detector tests."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.capture_workflow.blur_burst_detector import (
    BlurBurstReport,
    compute_blur_scores,
    detect_bursts,
    _laplacian_variance,
)


def _save_jpg(path: Path, image: np.ndarray):
    cv2.imwrite(str(path), image)


def _sharp(size=128, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def _blurred(size=128, seed=0):
    sharp = _sharp(size, seed)
    return cv2.GaussianBlur(sharp, (15, 15), 6.0)


def test_laplacian_variance_separates_sharp_from_blur():
    sharp = cv2.cvtColor(_sharp(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.cvtColor(_blurred(), cv2.COLOR_BGR2GRAY)
    assert _laplacian_variance(sharp) > _laplacian_variance(blurred) * 5


def test_compute_scores_unreadable_yields_zero(tmp_path):
    p = tmp_path / "missing.jpg"
    scores = compute_blur_scores([str(p)])
    assert scores == [0.0]


def test_detect_bursts_finds_3_consecutive_blurry(tmp_path):
    # 10 sharp + 4 blurry burst + 6 sharp
    paths = []
    for i in range(20):
        p = tmp_path / f"f_{i:03d}.jpg"
        if 10 <= i < 14:
            _save_jpg(p, _blurred(seed=i))
        else:
            _save_jpg(p, _sharp(seed=i))
        paths.append(str(p))

    rep = detect_bursts(paths, z_threshold=-1.0, min_burst_length=3)
    assert isinstance(rep, BlurBurstReport)
    assert rep.frame_count == 20
    # Detector should find at least one burst inside the 4-frame blur window
    assert len(rep.bursts) >= 1
    assert rep.total_burst_frames >= 3
    # The burst overlaps 10..13
    burst = rep.bursts[0]
    assert burst.start_index >= 10 and burst.end_index <= 13


def test_detect_bursts_short_run_under_threshold(tmp_path):
    # Single blurry frame surrounded by sharp — should not be flagged
    paths = []
    for i in range(10):
        p = tmp_path / f"f_{i:03d}.jpg"
        if i == 5:
            _save_jpg(p, _blurred(seed=i))
        else:
            _save_jpg(p, _sharp(seed=i))
        paths.append(str(p))

    rep = detect_bursts(paths, min_burst_length=3)
    # 1 frame run; min_burst_length=3 → no burst
    assert len(rep.bursts) == 0
    assert rep.total_burst_frames == 0


def test_detect_bursts_empty():
    rep = detect_bursts([])
    assert rep.frame_count == 0
    assert rep.bursts == []
    assert rep.burst_ratio == 0.0
    assert "no frames" in rep.notes


def test_detect_bursts_uniform_input_warns(tmp_path):
    # All identical frames → MAD≈0 → cannot detect bursts
    p_template = tmp_path / "tmpl.jpg"
    _save_jpg(p_template, _sharp(seed=42))
    paths = []
    for i in range(8):
        p = tmp_path / f"f_{i}.jpg"
        cv2.imwrite(str(p), cv2.imread(str(p_template)))
        paths.append(str(p))
    rep = detect_bursts(paths)
    assert any("MAD" in n for n in rep.notes)
    assert rep.bursts == []
