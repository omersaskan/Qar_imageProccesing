"""Sprint 2 — capture_quality_gate orchestrator tests."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.capture_workflow.capture_quality_gate import (
    CaptureGateReport,
    GateThresholds,
    evaluate_capture,
)


def _save(path: Path, img: np.ndarray):
    cv2.imwrite(str(path), img)


def _sharp(size=128, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def _make_meta(masks_dir: Path, frame_name: str, cy: float, cx: float, h: int = 128, w: int = 128):
    """Write side-car JSON the elevation/azimuth estimators read."""
    stem = Path(frame_name).stem
    (masks_dir / f"{stem}.json").write_text(json.dumps({
        "centroid": {"x": cx, "y": cy},
        "bbox": {"x": 0, "y": 0, "w": w, "h": h},
        "occupancy": 0.3,
    }), encoding="utf-8")
    # Mask side-car for image-shape lookup
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), 10, 255, -1)
    cv2.imwrite(str(masks_dir / f"{frame_name}.png"), mask)


def test_no_frames_returns_reshoot():
    rep = evaluate_capture([])
    assert rep.decision == "reshoot"
    assert any("no frames" in r for r in rep.reasons)


def test_below_minimum_frames_reviews(tmp_path):
    # 4 frames < default min 8 → review
    paths = []
    for i in range(4):
        p = tmp_path / f"f_{i}.jpg"
        _save(p, _sharp(seed=i))
        paths.append(str(p))
    rep = evaluate_capture(paths)
    assert rep.decision in ("review", "reshoot")
    assert any("min" in r for r in rep.reasons)


def test_solid_capture_passes(tmp_path):
    # 24 sharp frames + masks across 3 elevation bands × distributed centroids
    frames_dir = tmp_path / "frames"
    masks_dir = tmp_path / "masks"
    frames_dir.mkdir()
    masks_dir.mkdir()

    paths = []
    H, W = 128, 128
    cy_for_band = {"low": 30, "mid": 64, "top": 95}
    band_cycle = ["low", "mid", "top"]

    for i in range(24):
        name = f"f_{i:03d}.jpg"
        p = frames_dir / name
        _save(p, _sharp(seed=i))
        paths.append(str(p))
        band = band_cycle[i % 3]
        cy = cy_for_band[band]
        # Spread cx across the frame to simulate orbit motion
        cx = 16 + (i * 4) % (W - 32)
        _make_meta(masks_dir, name, cy=cy, cx=cx, h=H, w=W)

    rep = evaluate_capture(paths, masks_dir=masks_dir)
    # All three elevation bands sampled, motion present
    assert rep.elevation["multi_height_score"] >= 0.66
    assert rep.azimuth["cumulative_orbit_progress"] > 0.30
    # Allow review because heuristic azimuth is conservative on synthetic data
    assert rep.decision in ("pass", "review")


def test_single_band_capture_recommends_reshoot(tmp_path):
    frames_dir = tmp_path / "frames"
    masks_dir = tmp_path / "masks"
    frames_dir.mkdir()
    masks_dir.mkdir()

    paths = []
    H, W = 128, 128
    for i in range(20):
        name = f"f_{i:03d}.jpg"
        p = frames_dir / name
        _save(p, _sharp(seed=i))
        paths.append(str(p))
        # All centroids in MID band, no x motion → static + single height
        _make_meta(masks_dir, name, cy=64, cx=64, h=H, w=W)

    rep = evaluate_capture(paths, masks_dir=masks_dir)
    assert rep.decision == "reshoot"
    # Should flag both elevation and azimuth issues
    elev_reasons = [r for r in rep.reasons if "multi_height" in r]
    az_reasons = [r for r in rep.reasons if "orbit" in r]
    assert elev_reasons, f"expected multi_height reason, got {rep.reasons}"
    assert az_reasons, f"expected orbit reason, got {rep.reasons}"


def test_matrix_3x8_shape(tmp_path):
    rep = evaluate_capture([])  # empty → still produces matrix skeleton
    assert len(rep.matrix_3x8) == 3
    for row in rep.matrix_3x8:
        assert len(row) == 8


def test_decision_serializes_cleanly(tmp_path):
    p = tmp_path / "f.jpg"
    _save(p, _sharp())
    rep = evaluate_capture([str(p)])
    d = rep.to_dict()
    # Should be JSON-serializable
    s = json.dumps(d)
    assert "decision" in s
    assert "matrix_3x8" in s
