"""
Geometric quality metrics — manifoldness, hole area, normal consistency, etc.

These are "shape sanity" numbers; they don't replace visual QA but catch
catastrophic regressions: shattered mesh, flipped normals, collapsed
components, degenerate triangles.

All metrics return defensible values on empty / pathological input — no
exception bubbles up to the scorecard.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class GeometricReport:
    vertex_count: int = 0
    face_count: int = 0
    edge_count: int = 0

    is_watertight: bool = False
    is_winding_consistent: bool = False

    boundary_loop_count: int = 0
    non_manifold_edge_count: int = 0
    manifold_ratio: float = 1.0  # 1.0 = fully manifold

    hole_area_ratio: float = 0.0  # boundary loop area / total surface area

    edge_length_p50: float = 0.0
    edge_length_p99: float = 0.0
    edge_length_cv: float = 0.0  # coefficient of variation; lower = more uniform

    aspect_ratio_p50: float = 1.0
    aspect_ratio_p99: float = 1.0  # >100 → degenerate slivers present

    bbox_extent: Dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    surface_area: float = 0.0
    volume_to_bbox_ratio: float = 0.0  # compactness; sphere ≈ 0.52, cube = 1.0

    component_count: int = 0
    largest_component_face_share: float = 1.0

    grade: str = "F"           # A | B | C | F
    grade_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_split(mesh) -> int:
    try:
        comps = mesh.split(only_watertight=False)
        return max(1, len(comps))
    except Exception:
        return 1


def _largest_component_share(mesh) -> float:
    try:
        comps = mesh.split(only_watertight=False)
        if not comps:
            return 1.0
        sizes = [len(c.faces) for c in comps]
        return float(max(sizes) / max(sum(sizes), 1))
    except Exception:
        return 1.0


def _aspect_ratios(mesh) -> np.ndarray:
    """
    Per-triangle aspect ratio = longest_edge / shortest_edge.
    Slivers tend toward inf; equilateral = 1.
    """
    try:
        v = mesh.vertices
        f = mesh.faces
        a = v[f[:, 0]]
        b = v[f[:, 1]]
        c = v[f[:, 2]]
        e1 = np.linalg.norm(b - a, axis=1)
        e2 = np.linalg.norm(c - b, axis=1)
        e3 = np.linalg.norm(a - c, axis=1)
        edges = np.stack([e1, e2, e3], axis=1)
        long_e = np.max(edges, axis=1)
        short_e = np.min(edges, axis=1)
        ratios = np.where(short_e > 1e-9, long_e / short_e, np.inf)
        # Drop inf for percentile sanity
        finite = ratios[np.isfinite(ratios)]
        if finite.size == 0:
            return np.array([])
        return finite
    except Exception:
        return np.array([])


def _edge_lengths(mesh) -> np.ndarray:
    try:
        return mesh.edges_unique_length
    except Exception:
        return np.array([])


def _hole_area_ratio(mesh) -> float:
    """
    Approximate hole area as the bounding-area sum of boundary loops divided
    by total surface area.  Trimesh provides outline() for this.
    """
    try:
        if mesh.is_watertight:
            return 0.0
        outline = mesh.outline()
        if outline is None:
            return 0.0
        # Each entity is a Path; compute polygon-projected area
        total_hole = 0.0
        try:
            polys = outline.polygons_full
        except Exception:
            polys = []
        for poly in polys or []:
            try:
                total_hole += float(abs(poly.area))
            except Exception:
                continue
        sa = float(getattr(mesh, "area", 0.0))
        if sa <= 0:
            return 0.0
        return float(min(1.0, total_hole / sa))
    except Exception:
        return 0.0


def _grade(rep: "GeometricReport") -> tuple:
    """
    Letter grade from key signals.  Conservative — F for clearly broken meshes.
    """
    reasons = []
    if rep.face_count == 0 or rep.vertex_count == 0:
        return "F", ["empty mesh"]

    if rep.largest_component_face_share < 0.5:
        reasons.append(f"primary component only {rep.largest_component_face_share:.0%} of faces")
    if rep.hole_area_ratio > 0.30:
        reasons.append(f"hole_area_ratio {rep.hole_area_ratio:.0%} >30%")
    if rep.aspect_ratio_p99 > 200:
        reasons.append(f"aspect_ratio_p99 {rep.aspect_ratio_p99:.0f} (sliver triangles)")
    if rep.manifold_ratio < 0.85:
        reasons.append(f"manifold_ratio {rep.manifold_ratio:.0%} <85%")
    if not rep.is_winding_consistent:
        reasons.append("inconsistent face winding")

    if not reasons:
        return "A", []
    if len(reasons) == 1 and "manifold_ratio" not in reasons[0] and "inconsistent" not in reasons[0]:
        return "B", reasons
    if rep.face_count < 1000 or rep.largest_component_face_share < 0.30:
        return "F", reasons
    return "C", reasons


def compute_geometric_report(mesh: Optional[Any]) -> GeometricReport:
    """
    Compute geometric metrics for a trimesh.Trimesh.  Empty mesh → graded F
    but no exception.
    """
    rep = GeometricReport()
    if mesh is None:
        rep.grade_reasons = ["mesh is None"]
        return rep

    try:
        rep.vertex_count = int(len(mesh.vertices))
        rep.face_count = int(len(mesh.faces))
        rep.edge_count = int(len(mesh.edges_unique)) if hasattr(mesh, "edges_unique") else 0
    except Exception as e:
        rep.grade_reasons = [f"basic counts failed: {e}"]
        return rep

    if rep.face_count == 0:
        rep.grade_reasons = ["face_count == 0"]
        return rep

    # Topology booleans
    try:
        rep.is_watertight = bool(mesh.is_watertight)
    except Exception:
        rep.is_watertight = False
    try:
        rep.is_winding_consistent = bool(mesh.is_winding_consistent)
    except Exception:
        rep.is_winding_consistent = False

    # Boundary / manifold
    try:
        boundary_edges = mesh.edges[mesh.edges_unique_inverse][:, :].copy()  # placeholder
        # Trimesh ≥4: face_adjacency_unshared returns edges shared by <2 faces
        non_manifold = mesh.edges[mesh.edges_unique_length.argsort()][:0]  # default empty
        # Better: edges with degree != 2 are boundary or non-manifold
        try:
            edges_face_count = np.bincount(
                mesh.edges_unique_inverse, minlength=len(mesh.edges_unique)
            )
            non_manifold_count = int(np.sum(edges_face_count > 2))
            boundary_count = int(np.sum(edges_face_count == 1))
        except Exception:
            non_manifold_count = 0
            boundary_count = 0
        rep.non_manifold_edge_count = non_manifold_count
        rep.boundary_loop_count = boundary_count  # individual boundary edges; loop count is harder
        if rep.edge_count > 0:
            rep.manifold_ratio = float(1.0 - (non_manifold_count / rep.edge_count))
    except Exception:
        rep.manifold_ratio = 1.0

    # Hole area
    rep.hole_area_ratio = _hole_area_ratio(mesh)

    # Edge length stats
    el = _edge_lengths(mesh)
    if el.size > 0:
        rep.edge_length_p50 = float(np.median(el))
        rep.edge_length_p99 = float(np.percentile(el, 99))
        mean = float(np.mean(el))
        std = float(np.std(el))
        rep.edge_length_cv = float(std / mean) if mean > 1e-12 else 0.0

    # Aspect ratios
    ar = _aspect_ratios(mesh)
    if ar.size > 0:
        rep.aspect_ratio_p50 = float(np.median(ar))
        rep.aspect_ratio_p99 = float(np.percentile(ar, 99))

    # BBox / surface / volume
    try:
        bounds = mesh.bounds
        ext = bounds[1] - bounds[0]
        rep.bbox_extent = {"x": float(ext[0]), "y": float(ext[1]), "z": float(ext[2])}
        bbox_volume = float(np.prod(np.maximum(ext, 1e-9)))
    except Exception:
        bbox_volume = 0.0
    try:
        rep.surface_area = float(getattr(mesh, "area", 0.0))
    except Exception:
        pass
    try:
        if mesh.is_watertight:
            mesh_volume = float(abs(mesh.volume))
        else:
            mesh_volume = float(abs(mesh.convex_hull.volume))
        if bbox_volume > 1e-9:
            rep.volume_to_bbox_ratio = float(mesh_volume / bbox_volume)
    except Exception:
        pass

    # Components
    rep.component_count = _safe_split(mesh)
    rep.largest_component_face_share = _largest_component_share(mesh)

    # Grade
    grade, reasons = _grade(rep)
    rep.grade = grade
    rep.grade_reasons = reasons
    return rep
