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

    def _footprint_dominance(self, mesh: trimesh.Trimesh) -> float:
        bbox = self._bbox_extents(mesh)
        footprint = float(bbox[0] * bbox[1])
        height = float(max(bbox[2], 1e-8))
        return float(footprint / height)

    def _best_component_score(self, mesh: trimesh.Trimesh) -> float:
        components = mesh.split(only_watertight=False)
        if not components:
            return 0.0

        best_score = 0.0
        for comp in components:
            if len(comp.faces) < 50:
                continue
            best_score = max(best_score, float(self._score_component(comp)["total_score"]))

        if best_score <= 0.0:
            return float(self._score_component(mesh)["total_score"])
        return best_score

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

    def _remove_bottom_support_bands(
        self,
        mesh: trimesh.Trimesh,
        max_iter: int = 2,
    ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        """
        Removes large low-z support slabs even when they are not perfectly horizontal.

        This catches contamination that survives normal-based plane removal:
        - slightly tilted tables
        - thick support slabs
        - connected low support geometry around the base
        """
        working = mesh.copy()
        stats = {
            "removed_support_bands": 0,
            "removed_support_faces": 0,
            "removed_support_vertices": 0,
        }

        for _ in range(max_iter):
            if len(working.faces) == 0:
                break

            bounds = working.bounds
            extents = np.maximum(bounds[1] - bounds[0], 1e-8)
            total_height = float(extents[2])
            total_footprint = float(extents[0] * extents[1])
            if total_height <= 1e-8 or total_footprint <= 1e-8:
                break

            current_score = self._best_component_score(working)
            centers = working.triangles_center
            total_faces = max(len(working.faces), 1)
            best_candidate = None

            for frac in (0.12, 0.18, 0.24):
                z_cut = float(bounds[0][2] + total_height * frac)
                face_indices = np.where(centers[:, 2] <= z_cut)[0]
                if len(face_indices) < max(40, int(0.06 * total_faces)):
                    continue

                candidate_band = working.submesh([face_indices], append=True, repair=False)
                candidate_band = self._ensure_mesh(candidate_band)
                if len(candidate_band.faces) == 0:
                    continue

                band_extents = self._bbox_extents(candidate_band)
                footprint_ratio = float((band_extents[0] * band_extents[1]) / max(total_footprint, 1e-8))
                thickness_ratio = float(band_extents[2] / max(total_height, 1e-8))
                flatness_ratio = self._flatness_ratio(candidate_band)
                face_share = float(len(face_indices) / max(total_faces, 1))
                wide_band = bool(
                    band_extents[0] >= extents[0] * 0.55
                    or band_extents[1] >= extents[1] * 0.55
                )

                if not (
                    footprint_ratio >= 0.42
                    and thickness_ratio <= 0.22
                    and flatness_ratio <= 0.18
                    and face_share >= 0.08
                    and wide_band
                ):
                    continue

                trimmed = working.copy()
                keep_mask = np.ones(len(working.faces), dtype=bool)
                keep_mask[face_indices] = False
                trimmed.update_faces(keep_mask)
                trimmed.remove_unreferenced_vertices()

                if len(trimmed.faces) == 0:
                    continue

                trimmed_score = self._best_component_score(trimmed)
                score_gain = float(trimmed_score - current_score)
                candidate_quality = (
                    footprint_ratio * 0.35
                    + face_share * 0.20
                    + max(0.0, 0.25 - thickness_ratio) * 1.2
                    + max(0.0, score_gain) * 1.6
                )

                if best_candidate is None or candidate_quality > best_candidate["quality"]:
                    best_candidate = {
                        "face_indices": face_indices,
                        "quality": candidate_quality,
                        "score_gain": score_gain,
                    }

            if best_candidate is None:
                break

            if best_candidate["score_gain"] < 0.05 and len(best_candidate["face_indices"]) < int(0.18 * total_faces):
                break

            remove_mask = np.zeros(len(working.faces), dtype=bool)
            remove_mask[best_candidate["face_indices"]] = True
            removed_vertex_ids = np.unique(working.faces[remove_mask].reshape(-1))

            stats["removed_support_bands"] += 1
            stats["removed_support_faces"] += int(np.sum(remove_mask))
            stats["removed_support_vertices"] += int(len(removed_vertex_ids))

            working.update_faces(~remove_mask)
            working.remove_unreferenced_vertices()

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

        footprint_dominance = self._footprint_dominance(comp)
        footprint_penalty = 1.0 if footprint_dominance <= 10.0 else max(0.15, 10.0 / footprint_dominance)

        total_score = (
            face_score * 0.30
            + compactness_score * 0.20
            + flatness_penalty * 0.20
            + centrality_score * 0.15
            + aspect_penalty * 0.10
            + footprint_penalty * 0.05
        )

        return {
            "total_score": float(total_score),
            "face_score": float(face_score),
            "compactness_score": float(compactness),
            "flatness_score": float(flatness_ratio),
            "centrality_score": float(centrality_score),
            "aspect_penalty": float(aspect_penalty),
            "aspect_ratio": float(aspect_ratio),
            "footprint_dominance": float(footprint_dominance),
            "footprint_penalty": float(footprint_penalty),
        }

    def isolate_product(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, dict]:
        mesh = self._ensure_mesh(mesh)

        if len(mesh.vertices) == 0:
            return mesh, {"error": "Empty mesh"}

        initial_faces = int(len(mesh.faces))
        initial_vertices = int(len(mesh.vertices))

        # 1) remove dominant planes
        current_mesh, plane_stats = self._remove_horizontal_planes(mesh)
        current_mesh, support_stats = self._remove_bottom_support_bands(current_mesh)

        # 2) split components
        components = current_mesh.split(only_watertight=False)
        if not components:
            stats = {
                "initial_faces": initial_faces,
                "initial_vertices": initial_vertices,
                **plane_stats,
                **support_stats,
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
            **support_stats,
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
            "footprint_dominance": float(best_scores["footprint_dominance"]),
        }

        return best_comp, stats
