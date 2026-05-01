"""Sprint 1 — coverage_metrics tests."""
from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from modules.qa_validation.coverage_metrics import (
    compute_coverage_report,
    _azimuth_metrics,
    _elevation_metrics,
)


def test_azimuth_full_ring():
    yaws = [i * 45 + 5 for i in range(8)]  # one in each bucket
    m = _azimuth_metrics(yaws)
    assert m["azimuth_buckets_filled"] == 8
    assert m["azimuth_coverage_ratio"] == 1.0
    assert m["max_azimuth_gap_deg"] < 50


def test_azimuth_half_ring_has_large_gap():
    yaws = [0, 45, 90, 135]  # half ring
    m = _azimuth_metrics(yaws)
    assert m["azimuth_buckets_filled"] == 4
    assert m["azimuth_coverage_ratio"] == 0.5
    assert m["max_azimuth_gap_deg"] > 200


def test_azimuth_empty():
    m = _azimuth_metrics([])
    assert m["azimuth_buckets_filled"] == 0
    assert m["max_azimuth_gap_deg"] == 360.0


def test_elevation_buckets_and_multi_height():
    pitches = [-15, 10, 35, 60, 85]  # span all 5 buckets
    m = _elevation_metrics(pitches)
    assert m["elevation_buckets_filled"] == 5
    assert m["elevation_coverage_ratio"] == 1.0
    assert m["multi_height_score"] == 1.0  # low + mid + top all > 0
    assert m["multi_height_buckets"]["low"] >= 2
    assert m["multi_height_buckets"]["mid"] >= 1
    assert m["multi_height_buckets"]["top"] >= 1


def test_elevation_single_band_low_score():
    pitches = [10, 12, 15, 18]  # all in "low"
    m = _elevation_metrics(pitches)
    assert abs(m["multi_height_score"] - (1 / 3)) < 1e-6
    assert m["multi_height_buckets"]["mid"] == 0
    assert m["multi_height_buckets"]["top"] == 0


def test_compute_report_with_single_orbit_cameras():
    # Camera ring around origin at height=1, full 360°
    cams = []
    for i in range(8):
        ang = math.radians(i * 45)
        cams.append({"position": [math.cos(ang) * 2, math.sin(ang) * 2, 1.0]})
    rep = compute_coverage_report(cameras=cams)
    assert rep.azimuth_coverage_ratio == 1.0
    assert rep.azimuth_buckets_filled == 8
    # Single elevation band → multi_height ≈ 0.33
    assert rep.multi_height_score < 0.5
    assert rep.sample_count == 8


def test_compute_report_no_cameras_no_mesh():
    rep = compute_coverage_report()
    assert rep.sample_count == 0
    assert rep.observed_surface_ratio == 0.0
    assert rep.azimuth_coverage_ratio == 0.0


def test_compute_report_with_mesh_heuristic_observed():
    # No cameras / point cloud → observed surface falls back to heuristic = 1.0
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    rep = compute_coverage_report(mesh=mesh)
    assert rep.observed_surface_ratio == 1.0
    assert rep.observed_surface_method == "heuristic"


def test_dual_height_orbit_max_score():
    # Two rings: low and top.  Pitch is computed RELATIVE TO THE CAMERAS' OWN
    # CENTROID — so we have to make the elevation difference vs centroid clear.
    cams = []
    for i in range(8):
        ang = math.radians(i * 45)
        # Low ring far from centroid in z (well below)
        cams.append({"position": [math.cos(ang) * 3, math.sin(ang) * 3, -2.0]})
        # Top ring nearly above centroid (steep pitch)
        cams.append({"position": [math.cos(ang) * 0.3, math.sin(ang) * 0.3, 5.0]})
    rep = compute_coverage_report(cameras=cams)
    assert rep.azimuth_coverage_ratio == 1.0
    # Two distinct elevation bands → at least 2/3 multi_height score
    filled_buckets = sum(1 for v in rep.multi_height_buckets.values() if v > 0)
    assert filled_buckets >= 2, f"expected ≥2 filled height buckets, got {rep.multi_height_buckets}"
    assert rep.multi_height_score >= 0.66
