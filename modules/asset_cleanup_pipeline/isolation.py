import trimesh
import numpy as np
from typing import Tuple, Dict, Any, List
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("isolation")


class MeshIsolator:
    """
    Product isolation:
    - remove dominant horizontal-ish planes
    - split components
    - score components by product-likeness
    """

    def __init__(self):
        pass

    def _mesh_from_any(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        if isinstance(mesh, trimesh.Scene):
            return mesh.dump(concatenate=True)
        return mesh

    def _remove_dominant_planes(self, mesh: trimesh.Trimesh, max_iter: int = 2) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        working = mesh.copy()
        stats = {
            "removed_planes": 0,
            "removed_plane_faces": 0,
            "removed_plane_vertices": 0,
        }

        if len(working.faces) == 0:
            return working, stats

        for _ in range(max_iter):
            if len(working.faces) == 0:
                break

            centers = working.triangles_center
            normals = working.face_normals
            face_count_before = len(working.faces)
            vertex_count_before = len(working.vertices)

            # near-horizontal faces (table / floor tendency)
            horiz = np.abs(normals[:, 2]) > 0.92
            if horiz.sum() < max(20, int(0.05 * len(normals))):
                break

            z_vals = centers[horiz, 2]
            z_min, z_max = float(z_vals.min()), float(z_vals.max())
            span = max(1e-6, z_max - z_min)
            bins = max(10, min(60, int(len(z_vals) ** 0.5)))
            hist, edges = np.histogram(z_vals, bins=bins)
            peak_idx = int(np.argmax(hist))
            z_lo, z_hi = edges[peak_idx], edges[peak_idx + 1]
            z_mid = 0.5 * (z_lo + z_hi)
            z_tol = max(0.01, 0.03 * span)

            plane_faces = horiz & (np.abs(centers[:, 2] - z_mid) <= z_tol)
            removed_faces = int(plane_faces.sum())

            if removed_faces < max(30, int(0.08 * len(working.faces))):
                break

            plane_vertex_ids = np.unique(working.faces[plane_faces].reshape(-1))
            stats["removed_planes"] += 1
            stats["removed_plane_faces"] += removed_faces
            stats["removed_plane_vertices"] += int(len(plane_vertex_ids))

            keep_faces = ~plane_faces
            working.update_faces(keep_faces)
            working.remove_unreferenced_vertices()

            logger.info(f"Removed dominant plane candidate with {removed_faces} faces.")

            if len(working.faces) == face_count_before or len(working.vertices) == vertex_count_before:
                break

        return working, stats

    def _component_score(self, comp: trimesh.Trimesh) -> Dict[str, float]:
        faces = max(len(comp.faces), 1)
        bbox = comp.bounds[1] - comp.bounds[0]
        bbox = np.maximum(bbox, 1e-8)
        center_dist = float(np.linalg.norm(comp.centroid))

        max_dim = float(np.max(bbox))
        min_dim = float(np.min(bbox))
        flatness_ratio = min_dim / max(max_dim, 1e-8)

        bbox_volume = float(np.prod(bbox))
        try:
            comp_volume = float(abs(comp.volume)) if comp.is_watertight else float(comp.convex_hull.volume)
        except Exception:
            comp_volume = 0.0
        compactness = comp_volume / bbox_volume if bbox_volume > 1e-8 else 0.0

        face_score = min(1.0, np.log10(faces + 1) / 5.0)
        centrality_score = float(np.exp(-center_dist / 5.0))
        flatness_penalty = 1.0 if flatness_ratio >= 0.10 else (flatness_ratio / 0.10)
        compactness_score = min(1.0, compactness * 4.0)

        total = (
            face_score * 0.35
            + centrality_score * 0.20
            + flatness_penalty * 0.25
            + compactness_score * 0.20
        )

        return {
            "total_score": float(total),
            "face_score": float(face_score),
            "centrality_score": float(centrality_score),
            "flatness_score": float(flatness_ratio),
            "compactness_score": float(compactness),
        }

    def isolate_product(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, dict]:
        mesh = self._mesh_from_any(mesh)
        if len(mesh.vertices) == 0:
            return mesh, {"error": "Empty mesh"}

        initial_faces = len(mesh.faces)
        initial_vertices = len(mesh.vertices)

        current_mesh, plane_stats = self._remove_dominant_planes(mesh)

        components = current_mesh.split(only_watertight=False)
        if not components:
            stats = {
                "initial_faces": initial_faces,
                "initial_vertices": initial_vertices,
                **plane_stats,
                "component_count": 0,
                "removed_islands": 0,
                "final_faces": len(current_mesh.faces),
                "final_vertices": len(current_mesh.vertices),
                "removed_plane_face_share": plane_stats["removed_plane_faces"] / max(initial_faces, 1),
                "removed_plane_vertex_ratio": plane_stats["removed_plane_vertices"] / max(initial_vertices, 1),
            }
            return current_mesh, stats

        ranked: List[Tuple[float, trimesh.Trimesh, Dict[str, float]]] = []
        removed_islands = 0

        for comp in components:
            if len(comp.faces) < 100:
                removed_islands += 1
                continue
            scores = self._component_score(comp)
            ranked.append((scores["total_score"], comp, scores))

        if not ranked:
            best_comp = max(components, key=lambda c: len(c.faces))
            best_scores = self._component_score(best_comp)
        else:
            ranked.sort(key=lambda x: x[0], reverse=True)
            best_comp = ranked[0][1]
            best_scores = ranked[0][2]

        stats = {
            "initial_faces": initial_faces,
            "initial_vertices": initial_vertices,
            **plane_stats,
            "component_count": len(components),
            "removed_islands": removed_islands,
            "final_faces": len(best_comp.faces),
            "final_vertices": len(best_comp.vertices),
            "removed_plane_face_share": plane_stats["removed_plane_faces"] / max(initial_faces, 1),
            "removed_plane_vertex_ratio": plane_stats["removed_plane_vertices"] / max(initial_vertices, 1),
            "flatness_score": best_scores["flatness_score"],
            "compactness_score": best_scores["compactness_score"],
            "selected_component_score": best_scores["total_score"],
        }

        return best_comp, stats