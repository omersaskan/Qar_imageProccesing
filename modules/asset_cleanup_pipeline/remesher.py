import trimesh
import numpy as np
from typing import Tuple, Dict, Any, List
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("isolation")


class MeshIsolator:
    """
    Product isolation pipeline:
    1. remove dominant horizontal-ish planes (table/floor)
    2. split into connected components
    3. score components by product-likeness
    4. keep the best candidate
    """

    def __init__(self):
        pass

    def _ensure_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        return mesh

    def _bbox_extents(self, mesh: trimesh.Trimesh) -> np.ndarray:
        bounds = mesh.bounds
        return np.maximum(bounds[1] - bounds[0], 1e-8)

    def _compactness(self, mesh: trimesh.Trimesh) -> float:
        try:
            bbox = self._bbox_extents(mesh)
            bbox_volume = float(np.prod(bbox))
            if bbox_volume <= 1e-8:
                return 0.0

            if mesh.is_watertight:
                mesh_volume = float(abs(mesh.volume))
            else:
                mesh_volume = float(abs(mesh.convex_hull.volume))

            return float(mesh_volume / bbox_volume)
        except Exception:
            return 0.0

    def _flatness_ratio(self, mesh: trimesh.Trimesh) -> float:
        bbox = self._bbox_extents(mesh)
        return float(np.min(bbox) / np.max(bbox))

    def _remove_horizontal_planes(
        self,
        mesh: trimesh.Trimesh,
        max_iter: int = 2,
    ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        """
        Removes large horizontal-ish face groups, which are likely table/floor planes.

        Strategy:
        - find near-horizontal faces using face normals
        - histogram their z-centers
        - remove dominant z-slab if sufficiently large
        """
        working = mesh.copy()

        stats = {
            "removed_planes": 0,
            "removed_plane_faces": 0,
            "removed_plane_vertices": 0,
            "plane_candidate_count": 0,
        }

        if len(working.faces) == 0:
            return working, stats

        for _ in range(max_iter):
            if len(working.faces) == 0:
                break

            centers = working.triangles_center
            normals = working.face_normals

            # near-horizontal faces → tables/floors are usually aligned like this
            horizontal_faces = np.abs(normals[:, 2]) > 0.92
            horiz_count = int(np.sum(horizontal_faces))

            if horiz_count < max(20, int(0.05 * len(working.faces))):
                break

            stats["plane_candidate_count"] += 1

            z_vals = centers[horizontal_faces, 2]
            z_min, z_max = float(z_vals.min()), float(z_vals.max())
            span = max(z_max - z_min, 1e-6)

            bin_count = max(10, min(60, int(np.sqrt(len(z_vals)))))
            hist, edges = np.histogram(z_vals, bins=bin_count)
            peak_idx = int(np.argmax(hist))
            z_lo, z_hi = edges[peak_idx], edges[peak_idx + 1]
            z_mid = 0.5 * (z_lo + z_hi)

            z_tol = max(0.01, 0.03 * span)
            plane_faces = horizontal_faces & (np.abs(centers[:, 2] - z_mid) <= z_tol)

            removed_faces = int(np.sum(plane_faces))
            if removed_faces < max(30, int(0.08 * len(working.faces))):
                break

            plane_vertex_ids = np.unique(working.faces[plane_faces].reshape(-1))
            stats["removed_planes"] += 1
            stats["removed_plane_faces"] += removed_faces
            stats["removed_plane_vertices"] += int(len(plane_vertex_ids))

            keep_faces = ~plane_faces
            working.update_faces(keep_faces)
            working.remove_unreferenced_vertices()

            logger.info(
                f"Removed dominant plane candidate: faces={removed_faces}, "
                f"remaining_faces={len(working.faces)}"
            )

        return working, stats

    def _score_component(self, comp: trimesh.Trimesh) -> Dict[str, float]:
        faces = max(len(comp.faces), 1)
        bbox = self._bbox_extents(comp)

        center_dist = float(np.linalg.norm(comp.centroid))
        face_score = min(1.0, np.log10(faces + 1) / 5.0)

        compactness = self._compactness(comp)
        compactness_score = min(1.0, compactness * 4.0)

        flatness_ratio = self._flatness_ratio(comp)
        flatness_penalty = 1.0 if flatness_ratio >= 0.10 else (flatness_ratio / 0.10)

        centrality_score = float(np.exp(-center_dist / 5.0))

        aspect_ratio = float(np.max(bbox) / max(np.min(bbox), 1e-8))
        aspect_penalty = 1.0 if aspect_ratio <= 12.0 else max(0.2, 12.0 / aspect_ratio)

        total_score = (
            face_score * 0.30
            + compactness_score * 0.20
            + flatness_penalty * 0.20
            + centrality_score * 0.15
            + aspect_penalty * 0.15
        )

        return {
            "total_score": float(total_score),
            "face_score": float(face_score),
            "compactness_score": float(compactness),
            "flatness_score": float(flatness_ratio),
            "centrality_score": float(centrality_score),
            "aspect_penalty": float(aspect_penalty),
            "aspect_ratio": float(aspect_ratio),
        }

    def isolate_product(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, dict]:
        mesh = self._ensure_mesh(mesh)

        if len(mesh.vertices) == 0:
            return mesh, {"error": "Empty mesh"}

        initial_faces = int(len(mesh.faces))
        initial_vertices = int(len(mesh.vertices))

        # 1) remove dominant planes
        current_mesh, plane_stats = self._remove_horizontal_planes(mesh)

        # 2) split components
        components = current_mesh.split(only_watertight=False)
        if not components:
            stats = {
                "initial_faces": initial_faces,
                "initial_vertices": initial_vertices,
                **plane_stats,
                "component_count": 0,
                "removed_islands": 0,
                "final_faces": int(len(current_mesh.faces)),
                "final_vertices": int(len(current_mesh.vertices)),
                "removed_plane_face_share": plane_stats["removed_plane_faces"] / max(initial_faces, 1),
                "removed_plane_vertex_ratio": plane_stats["removed_plane_vertices"] / max(initial_vertices, 1),
                "compactness_score": 0.0,
                "flatness_score": 0.0,
                "selected_component_score": 0.0,
            }
            return current_mesh, stats

        ranked: List[Tuple[float, trimesh.Trimesh, Dict[str, float]]] = []
        removed_islands = 0

        for comp in components:
            if len(comp.faces) < 100:
                removed_islands += 1
                continue

            scores = self._score_component(comp)
            ranked.append((scores["total_score"], comp, scores))

        if not ranked:
            best_comp = max(components, key=lambda c: len(c.faces))
            best_scores = self._score_component(best_comp)
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
            "final_faces": int(len(best_comp.faces)),
            "final_vertices": int(len(best_comp.vertices)),
            "removed_plane_face_share": plane_stats["removed_plane_faces"] / max(initial_faces, 1),
            "removed_plane_vertex_ratio": plane_stats["removed_plane_vertices"] / max(initial_vertices, 1),
            "compactness_score": float(best_scores["compactness_score"]),
            "flatness_score": float(best_scores["flatness_score"]),
            "selected_component_score": float(best_scores["total_score"]),
            "aspect_ratio": float(best_scores["aspect_ratio"]),
        }

        return best_comp, stats