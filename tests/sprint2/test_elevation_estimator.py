"""Sprint 2 — elevation_estimator tests."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.capture_workflow.elevation_estimator import (
    ElevationReport,
    estimate_elevation_distribution,
    _bucket_for,
)


def test_bucket_thresholds():
    assert _bucket_for(0.20) == "low"   # object near top of frame → looking up
    assert _bucket_for(0.50) == "mid"
    assert _bucket_for(0.80) == "top"   # object near bottom → looking down


def _write_meta_and_mask(masks_dir: Path, name: str, cy: float, h: int = 128, w: int = 128):
    stem = Path(name).stem
    (masks_dir / f"{stem}.json").write_text(json.dumps({
        "centroid": {"x": w / 2, "y": cy},
    }), encoding="utf-8")
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, int(cy)), 10, 255, -1)
    cv2.imwrite(str(masks_dir / f"{name}.png"), mask)


def test_no_frames_returns_empty_report():
    rep = estimate_elevation_distribution([])
    assert rep.frame_count == 0
    assert rep.multi_height_score == 0.0


def test_no_masks_dir_warns(tmp_path):
    p = tmp_path / "f.jpg"
    p.touch()
    rep = estimate_elevation_distribution([str(p)], masks_dir=None)
    assert rep.frame_count == 1
    assert any("masks_dir" in n for n in rep.notes)


def test_single_band_capture(tmp_path):
    frames_dir = tmp_path / "frames"
    masks_dir = tmp_path / "masks"
    frames_dir.mkdir()
    masks_dir.mkdir()

    paths = []
    for i in range(6):
        name = f"f_{i}.jpg"
        (frames_dir / name).touch()
        _write_meta_and_mask(masks_dir, name, cy=64)  # all mid
        paths.append(str(frames_dir / name))

    rep = estimate_elevation_distribution(paths, masks_dir=masks_dir)
    assert rep.bucket_counts["mid"] == 6
    assert rep.bucket_counts["low"] == 0
    assert rep.bucket_counts["top"] == 0
    assert abs(rep.multi_height_score - (1 / 3)) < 1e-6
    assert any("one elevation band" in n for n in rep.notes)


def test_three_band_capture(tmp_path):
    frames_dir = tmp_path / "frames"
    masks_dir = tmp_path / "masks"
    frames_dir.mkdir()
    masks_dir.mkdir()

    paths = []
    bands = [25, 64, 100]  # low cy → "low" bucket; mid; high cy → "top"
    for i in range(9):
        name = f"f_{i}.jpg"
        (frames_dir / name).touch()
        _write_meta_and_mask(masks_dir, name, cy=bands[i % 3])
        paths.append(str(frames_dir / name))

    rep = estimate_elevation_distribution(paths, masks_dir=masks_dir)
    assert rep.bucket_counts["low"] >= 1
    assert rep.bucket_counts["mid"] >= 1
    assert rep.bucket_counts["top"] >= 1
    assert rep.multi_height_score == 1.0


def test_two_band_capture_warns(tmp_path):
    frames_dir = tmp_path / "frames"
    masks_dir = tmp_path / "masks"
    frames_dir.mkdir()
    masks_dir.mkdir()

    paths = []
    for i in range(8):
        name = f"f_{i}.jpg"
        (frames_dir / name).touch()
        cy = 25 if i % 2 == 0 else 64  # low + mid
        _write_meta_and_mask(masks_dir, name, cy=cy)
        paths.append(str(frames_dir / name))

    rep = estimate_elevation_distribution(paths, masks_dir=masks_dir)
    assert abs(rep.multi_height_score - (2 / 3)) < 1e-6
    assert any("third is missing" in n for n in rep.notes)
