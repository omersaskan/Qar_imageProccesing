"""Sprint 1 — geometric_quality tests."""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from modules.qa_validation.geometric_quality import compute_geometric_report


def test_clean_box_grades_a():
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    rep = compute_geometric_report(mesh)
    assert rep.face_count == 12
    assert rep.is_watertight is True
    assert rep.is_winding_consistent is True
    assert rep.component_count == 1
    assert rep.largest_component_face_share == 1.0
    assert rep.grade == "A"
    assert rep.hole_area_ratio == 0.0
    assert rep.aspect_ratio_p99 < 5.0  # box triangles are well-shaped


def test_empty_mesh_grades_f():
    rep = compute_geometric_report(None)
    assert rep.grade == "F"


def test_three_equal_components_demote_grade():
    # Three identical disjoint boxes: each is 12 faces → primary share = 0.33,
    # which crosses the <0.5 demotion threshold for sure.
    boxes = []
    for i, off in enumerate([0, 5, 10]):
        b = trimesh.creation.box(extents=[1, 1, 1])
        b.apply_translation([off, 0, 0])
        boxes.append(b)
    combined = trimesh.util.concatenate(boxes)
    rep = compute_geometric_report(combined)
    assert rep.component_count >= 3
    assert rep.largest_component_face_share < 0.5
    assert rep.grade in ("B", "C", "F"), f"Expected demoted grade, got {rep.grade}"


def test_ico_sphere_compactness():
    sphere = trimesh.creation.icosphere(subdivisions=2)
    rep = compute_geometric_report(sphere)
    assert rep.is_watertight is True
    # Sphere/bbox ratio ≈ 0.524
    assert 0.45 < rep.volume_to_bbox_ratio < 0.62
    assert rep.grade == "A"


def test_open_mesh_has_holes():
    # Take half a box → open mesh
    box = trimesh.creation.box(extents=[1, 1, 1])
    # Remove the top face (face indices 4 and 5 typically)
    keep = list(range(len(box.faces)))
    keep.remove(0)
    keep.remove(1)
    open_mesh = box.submesh([keep], append=True)
    rep = compute_geometric_report(open_mesh)
    assert rep.is_watertight is False
    # boundary edges > 0 expected
    assert rep.boundary_loop_count > 0


def test_metrics_keys_present():
    """Schema sanity — every documented metric key exists."""
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    rep = compute_geometric_report(mesh)
    d = rep.to_dict()
    required = {
        "vertex_count", "face_count", "edge_count",
        "is_watertight", "is_winding_consistent",
        "boundary_loop_count", "non_manifold_edge_count", "manifold_ratio",
        "hole_area_ratio",
        "edge_length_p50", "edge_length_p99", "edge_length_cv",
        "aspect_ratio_p50", "aspect_ratio_p99",
        "bbox_extent", "surface_area", "volume_to_bbox_ratio",
        "component_count", "largest_component_face_share",
        "grade", "grade_reasons",
    }
    assert required.issubset(set(d.keys()))
