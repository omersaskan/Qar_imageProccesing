import trimesh
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from modules.operations.logging_config import get_component_logger
from .camera_projection import compute_component_mask_support

logger = get_component_logger("isolation")


class MeshIsolator:
    """
    Product isolation pipeline:
    1. remove dominant horizontal-ish planes (table/floor)
    2. split into connected components
    3. score components by:
       - product-likeness (geometric)
       - mask support (semantic)
       - point cloud support (reconstruction confidence)
    4. keep the best candidates
    """

    def __init__(self):
        pass

    def _try_load_background_rgb(self, output_dir: Path) -> Optional[Tuple[int, int, int]]:
        """
        Walk up from `output_dir` looking for `extraction_manifest.json` (max 4 levels)
        and pull `color_profile.background_rgb` if present.
        Returns None when no manifest / no profile / malformed.
        """
        try:
            search_roots = [output_dir]
            search_roots.extend(list(output_dir.parents)[:4])
            for root in search_roots:
                if not root.exists():
                    continue
                # Direct hit
                direct = root / "extraction_manifest.json"
                if direct.exists():
                    candidates = [direct]
                else:
                    # Shallow rglob (max 3 deep)
                    candidates = []
                    for c in root.rglob("extraction_manifest.json"):
                        try:
                            depth = len(c.relative_to(root).parts)
                        except ValueError:
                            continue
                        if depth <= 3:
                            candidates.append(c)
                        if len(candidates) >= 3:
                            break
                for cand in candidates:
                    try:
                        with open(cand, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        cp = data.get("color_profile") or {}
                        bg = cp.get("background_rgb")
                        if isinstance(bg, (list, tuple)) and len(bg) >= 3:
                            return (int(bg[0]), int(bg[1]), int(bg[2]))
                    except Exception:
                        continue
        except Exception:
            pass
        return None

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

    def _remove_horizontal_planes_tracked(
        self,
        mesh: trimesh.Trimesh,
        max_iter: int = 2,
    ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        working = mesh.copy()
        if 'face_indices' not in working.metadata:
            working.metadata['face_indices'] = np.arange(len(working.faces))

        stats = {
            "removed_planes": 0,
            "removed_plane_faces": 0,
            "removed_plane_vertices": 0,
            "plane_candidate_count": 0,
        }

        for _ in range(max_iter):
            if len(working.faces) == 0:
                break

            centers = working.triangles_center
            normals = working.face_normals
            horizontal_faces = np.abs(normals[:, 2]) > 0.92
            horiz_count = int(np.sum(horizontal_faces))

            if horiz_count < max(20, int(0.05 * len(working.faces))):
                break

            stats["plane_candidate_count"] += 1
            z_vals = centers[horizontal_faces, 2]
            z_mid = 0.5 * (float(z_vals.min()) + float(z_vals.max()))
            z_tol = max(0.01, 0.03 * (float(z_vals.max()) - float(z_vals.min())))
            plane_faces = horizontal_faces & (np.abs(centers[:, 2] - z_mid) <= z_tol)

            removed_faces = int(np.sum(plane_faces))
            if removed_faces < max(30, int(0.08 * len(working.faces))):
                break

            stats["removed_planes"] += 1
            stats["removed_plane_faces"] += removed_faces
            
            indices_to_keep = np.where(~plane_faces)[0]
            original_indices = working.metadata['face_indices'][indices_to_keep]
            working = working.submesh([indices_to_keep], append=True)
            working.metadata['face_indices'] = original_indices

        return working, stats

    def _remove_bottom_support_bands_tracked(
        self,
        mesh: trimesh.Trimesh,
        max_iter: int = 2,
    ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        working = mesh.copy()
        if 'face_indices' not in working.metadata:
            working.metadata['face_indices'] = np.arange(len(working.faces))

        stats = {
            "removed_support_bands": 0,
            "removed_support_faces": 0,
            "removed_support_vertices": 0,
        }

        for _ in range(max_iter):
            if len(working.faces) == 0:
                break

            bounds = working.bounds
            total_height = float(np.maximum(bounds[1] - bounds[0], 1e-8)[2])
            total_footprint = float(np.prod(np.maximum(bounds[1] - bounds[0], 1e-8)[:2]))
            
            current_score = self._best_component_score(working)
            centers = working.triangles_center
            best_candidate = None

            for frac in (0.12, 0.18, 0.24):
                z_cut = float(bounds[0][2] + total_height * frac)
                face_indices = np.where(centers[:, 2] <= z_cut)[0]
                if len(face_indices) < max(40, int(0.06 * len(working.faces))):
                    continue

                candidate_band = working.submesh([face_indices], append=True)
                if len(candidate_band.faces) == 0: continue

                band_extents = self._bbox_extents(candidate_band)
                footprint_ratio = float((band_extents[0] * band_extents[1]) / max(total_footprint, 1e-8))
                thickness_ratio = float(band_extents[2] / max(total_height, 1e-8))
                
                if not (footprint_ratio >= 0.42 and thickness_ratio <= 0.22):
                    continue

                remaining_indices = np.delete(np.arange(len(working.faces)), face_indices)
                remaining_mesh = working.submesh([remaining_indices], append=True)
                
                after_score = self._best_component_score(remaining_mesh)
                if after_score > current_score * 0.95:
                    best_candidate = face_indices
                    break

            if best_candidate is not None:
                stats["removed_support_bands"] += 1
                stats["removed_support_faces"] += len(best_candidate)
                
                indices_to_keep = np.delete(np.arange(len(working.faces)), best_candidate)
                original_indices = working.metadata['face_indices'][indices_to_keep]
                working = working.submesh([indices_to_keep], append=True)
                working.metadata['face_indices'] = original_indices
            else:
                break

        return working, stats

    def _score_component(
        self, 
        comp: trimesh.Trimesh, 
        mask_support: Optional[Dict] = None, 
        pc_support: Optional[Dict] = None
    ) -> Dict[str, float]:
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

        # Base geometric score (0.0 to 1.0)
        geom_score = (
            face_score * 0.30
            + compactness_score * 0.20
            + flatness_penalty * 0.20
            + centrality_score * 0.15
            + aspect_penalty * 0.10
            + footprint_penalty * 0.05
        )

        total_score = geom_score
        
        # Integration of Semantic Support (Masks)
        mask_score = 0.0
        if mask_support:
            # We use a mix of average support and supported view count
            avg_support = mask_support.get("avg_support", 0.0)
            hit_ratio = mask_support.get("hit_ratio", 0.0)
            view_count = mask_support.get("view_count", 1)
            supported_views = mask_support.get("supported_view_count", 0)
            
            # Semantic confidence is high if avg support is high AND it's visible in many views
            mask_score = (avg_support * 0.7 + hit_ratio * 0.3)
            
            # Boost score if we have semantic confirmation
            # If mask support is very high (>0.8), it's almost certainly the product
            # If mask support is very low (<0.2), it's almost certainly NOT the product
            total_score = total_score * 0.4 + mask_score * 0.6
            
            if supported_views < 3 and view_count > 10:
                # Penalty for components that are mostly outside masks in many views
                total_score *= 0.5

        # Integration of Point Cloud Support
        pc_score = 0.0
        if pc_support:
            pc_score = pc_support.get("support_ratio", 0.0)
            # PC support confirms the geometry exists in the dense reconstruction
            # If we have both, we weigh PC heavily as it's the "ground truth" of the reconstruction
            if mask_support:
                total_score = total_score * 0.7 + pc_score * 0.3
            else:
                total_score = total_score * 0.6 + pc_score * 0.4

        return {
            "total_score": float(total_score),
            "geom_score": float(geom_score),
            "mask_score": float(mask_score),
            "pc_score": float(pc_score),
            "face_score": float(face_score),
            "compactness_score": float(compactness),
            "flatness_score": float(flatness_ratio),
            "centrality_score": float(centrality_score),
            "aspect_penalty": float(aspect_penalty),
            "aspect_ratio": float(aspect_ratio),
            "footprint_dominance": float(footprint_dominance),
            "footprint_penalty": float(footprint_penalty),
        }

    def isolate_by_point_cloud(
        self,
        mesh: trimesh.Trimesh,
        point_cloud: trimesh.points.PointCloud,
        dist_threshold: float = 0.05,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Computes support for each component from the dense point cloud.
        """
        if point_cloud is None or len(point_cloud.vertices) == 0:
            return {}

        components = mesh.split(only_watertight=False)
        if not components:
            return {}

        from scipy.spatial import cKDTree
        tree = cKDTree(point_cloud.vertices)

        results = {}
        for i, comp in enumerate(components):
            if len(comp.vertices) == 0: continue
            
            dists, _ = tree.query(comp.vertices, k=1)
            supported = np.sum(dists < dist_threshold)
            support_ratio = supported / len(comp.vertices)
            
            results[i] = {
                "supported_vertices": int(supported),
                "total_vertices": len(comp.vertices),
                "support_ratio": float(support_ratio)
            }
            
        return results

    def isolate_product(
        self,
        mesh: trimesh.Trimesh,
        cameras: Optional[List[Dict]] = None,
        masks: Optional[Dict[str, np.ndarray]] = None,
        point_cloud: Optional[trimesh.points.PointCloud] = None,
        output_dir: Optional[Path] = None,
        background_rgb: Optional[Tuple[int, int, int]] = None,
        background_tolerance: int = 35,
    ) -> Tuple[trimesh.Trimesh, dict]:
        # SPRINT 5 Fix: Preserve original visuals to re-apply after isolation.
        # Isolation processing (splitting, plane removal) will use a geom-only copy 
        # to avoid MemoryError with large textures.
        original_visual = None
        if hasattr(mesh, 'visual') and mesh.visual is not None:
            original_visual = mesh.visual

        mesh = self._ensure_mesh(mesh)
        
        # Create a geometry-only working copy
        working_geom = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
        working_geom.metadata['face_indices'] = np.arange(len(mesh.faces))
        
        if isinstance(mesh, trimesh.points.PointCloud):
            return mesh, {"error": "Input is a point cloud, expected a mesh with faces", "object_isolation_status": "failed"}

        if len(mesh.vertices) == 0:
            return mesh, {"error": "Empty mesh", "object_isolation_status": "failed"}

        initial_faces = int(len(mesh.faces))
        initial_vertices = int(len(mesh.vertices))
        
        isolation_method = "geometric_only"
        reason_if_geometric_fallback = None
        
        if not cameras or not masks:
            reason_if_geometric_fallback = "No cameras or masks provided"
        elif not cameras:
            reason_if_geometric_fallback = "Cameras missing"
        elif not masks:
            reason_if_geometric_fallback = "Masks missing"

        # Auto-resolve background_rgb from extraction manifest if caller didn't pass one.
        if background_rgb is None and output_dir is not None:
            background_rgb = self._try_load_background_rgb(output_dir)

        # 1) remove dominant planes (Geometric pre-pass)
        # We need to track face indices. Let's update _remove_horizontal_planes to use submesh.
        current_mesh, plane_stats = self._remove_horizontal_planes_tracked(working_geom)
        current_mesh, support_stats = self._remove_bottom_support_bands_tracked(current_mesh)
        
        # 1.5) Chromatic Leakage Removal (generic background filter; legacy orange fallback if no bg_rgb)
        current_mesh, chromatic_stats = self._remove_chromatic_leakage_tracked(
            current_mesh, mesh,
            background_rgb=background_rgb,
            background_tolerance=background_tolerance,
        )

        # 2) split components
        # Optimization: trimesh.split() doesn't preserve metadata, so we do it manually
        if 'face_indices' in current_mesh.metadata:
            # Get face masks for components
            from trimesh.graph import connected_components
            face_groups = connected_components(current_mesh.face_adjacency, nodes=np.arange(len(current_mesh.faces)))
            components = []
            for group in face_groups:
                if len(group) == 0: continue
                comp = current_mesh.submesh([group], append=True)
                comp.metadata['face_indices'] = current_mesh.metadata['face_indices'][group]
                components.append(comp)
        else:
            components = current_mesh.split(only_watertight=False)

        if not components:
             return current_mesh, {"object_isolation_status": "failed_no_components", "initial_faces": initial_faces}

        # Save debug components before isolation
        if output_dir:
            try:
                current_mesh.export(str(output_dir / "debug_components_before_isolation.obj"))
            except Exception: pass

        # 3) Compute Support Metrics
        mask_supports = {}
        if cameras and masks:
            isolation_method = "mask_guided"
            for i, comp in enumerate(components):
                if len(comp.faces) < 20: continue
                mask_supports[i] = compute_component_mask_support(comp, cameras, masks)
            
            if output_dir:
                with open(output_dir / "mask_projection_report.json", "w") as f:
                    json.dump(mask_supports, f, indent=2)

        pc_supports = {}
        if point_cloud is not None:
            pc_supports = self.isolate_by_point_cloud(current_mesh, point_cloud)
            if isolation_method == "mask_guided":
                isolation_method = "hybrid_pc_mask"
            else:
                isolation_method = "pc_guided"

        # 4) Score and Rank
        ranked = []
        all_scores = {}
        for i, comp in enumerate(components):
            if len(comp.faces) < 50:
                continue
            
            scores = self._score_component(
                comp, 
                mask_support=mask_supports.get(i), 
                pc_support=pc_supports.get(i)
            )
            ranked.append((scores["total_score"], i, comp, scores))
            all_scores[i] = {
                "faces": len(comp.faces),
                "geometric_score": scores["geom_score"],
                "mask_support": mask_supports.get(i, {}),
                "pc_support": pc_supports.get(i, {}),
                "total_score": scores["total_score"],
                "decision": "pending"
            }
            
        # Detect if SAM2 was used in any mask (experimental flag for gating)
        used_sam2 = False
        # Try to find masks dir relative to output_dir (cleanup) or assume it's provided in metadata if available
        potential_mask_dirs = []
        if output_dir:
            # Check for data/captures/<job_id>/frames/masks
            # job_id is usually extracted from output_dir name
            job_id = output_dir.name
            if job_id.endswith("_sam2_experiment"):
                job_id = job_id.replace("_sam2_experiment", "")
            
            # Common structure: data/captures/<id>/frames/masks
            potential_mask_dirs.append(output_dir.parent.parent / "captures" / job_id / "frames" / "masks")
            potential_mask_dirs.append(output_dir.parent / "frames" / "masks") # direct parent
        
        for mask_meta_dir in potential_mask_dirs:
            if mask_meta_dir.exists():
                for meta_file in mask_meta_dir.glob("*.json"):
                    try:
                        with open(meta_file, "r") as f:
                            if json.load(f).get("segmentation_method") == "sam2":
                                used_sam2 = True
                                break
                    except Exception: pass
                if used_sam2: break

        if not ranked:
            # Absolute fallback: keep largest
            best_idx = int(np.argmax([len(c.faces) for c in components]))
            best_comp = components[best_idx]
            kept_components = [best_comp]
            best_scores = self._score_component(best_comp)
            if best_idx not in all_scores:
                all_scores[best_idx] = {
                    "faces": len(best_comp.faces),
                    "geometric_score": best_scores["geom_score"],
                    "total_score": best_scores["total_score"],
                }
            all_scores[best_idx]["decision"] = "kept_fallback_largest"
        else:
            ranked.sort(key=lambda x: x[0], reverse=True)
            
            best_score = ranked[0][0]
            best_faces = len(ranked[0][2].faces)
            total_faces_ranked = sum(len(r[2].faces) for r in ranked)
            
            # SPRINT Hardening: primary_assignment guard
            # If top-score component is a tiny fragment (< 5% of ranked faces), 
            # check if there's a significantly larger component that is also strong.
            primary_idx_in_ranked = 0
            primary_assignment_result = "normal"
            
            if best_faces < total_faces_ranked * 0.05:
                # Find largest component
                largest_idx_in_ranked = int(np.argmax([len(r[2].faces) for r in ranked]))
                largest_faces = len(ranked[largest_idx_in_ranked][2].faces)
                largest_score = ranked[largest_idx_in_ranked][0]
                
                if largest_faces > best_faces * 10 and largest_score > 0.40:
                    # Conflict! Tiny fragment has higher score but large component exists.
                    primary_assignment_result = "primary_assignment_conflict"
                    logger.warning(f"Primary assignment conflict: Tiny component (idx={ranked[0][1]}, faces={best_faces}) has higher score than main body (idx={ranked[largest_idx_in_ranked][1]}, faces={largest_faces}). Swapping primary.")
                    primary_idx_in_ranked = largest_idx_in_ranked
            
            primary_comp = ranked[primary_idx_in_ranked][2]
            best_scores = ranked[primary_idx_in_ranked][3]
            best_score = ranked[primary_idx_in_ranked][0]
            best_faces = len(primary_comp.faces)
            kept_components = [primary_comp]
            all_scores[ranked[primary_idx_in_ranked][1]]["decision"] = "kept_primary"
            
            # Keep secondary components if they are strong
            # We are more strict if we have semantic guidance
            threshold_ratio = 0.75 if isolation_method != "geometric_only" else 0.70
            
            for i, (score, idx, comp, s) in enumerate(ranked):
                if i == primary_idx_in_ranked: continue # Skip primary
                
                if score > best_score * threshold_ratio and len(comp.faces) > best_faces * 0.05:
                    kept_components.append(comp)
                    all_scores[idx]["decision"] = "kept_secondary"
                else:
                    all_scores[idx]["decision"] = "rejected_low_score"

        if output_dir:
            with open(output_dir / "component_scores.json", "w") as f:
                json.dump(all_scores, f, indent=2)

        # Concatenate kept components
        final_geom = trimesh.util.concatenate(kept_components) if len(kept_components) > 1 else kept_components[0]
        
        # Re-apply visuals if they existed
        if original_visual is not None and 'face_indices' in final_geom.metadata:
            # trimesh.util.concatenate handles metadata if we are lucky, but let's be safe
            all_face_indices = []
            for comp in kept_components:
                if 'face_indices' in comp.metadata:
                    all_face_indices.extend(comp.metadata['face_indices'])
            
            if all_face_indices:
                final_mesh = trimesh.Trimesh(vertices=final_geom.vertices, faces=final_geom.faces, process=False)
                final_mesh.visual = original_visual.face_subset(all_face_indices)
            else:
                final_mesh = final_geom
        else:
            final_mesh = final_geom
        final_faces = int(len(final_mesh.faces))
        removed_face_ratio = (initial_faces - final_faces) / max(initial_faces, 1)

        # Compute detailed quality metrics for post-cleanup gating
        primary_faces = int(len(kept_components[0].faces))
        primary_face_share = primary_faces / max(final_faces, 1)
        kept_to_initial_face_ratio = final_faces / max(initial_faces, 1)
        
        # Largest kept component (may differ from primary if sorted differently)
        largest_kept_faces = max(len(c.faces) for c in kept_components)
        largest_kept_component_share = largest_kept_faces / max(final_faces, 1)

        # Final Summary Metrics
        stats = {
            "initial_faces": initial_faces,
            "initial_vertices": initial_vertices,
            **plane_stats,
            **support_stats,
            **chromatic_stats,
            "kept_component_count": len(kept_components),
            "raw_component_count": len(components),
            "rejected_component_count": len(components) - len(kept_components),
            "final_faces": final_faces,
            "final_vertices": int(len(final_mesh.vertices)),
            "removed_face_ratio": float(removed_face_ratio),
            "primary_component_faces": primary_faces,
            "primary_face_share": float(primary_face_share),
            "total_kept_faces": final_faces,
            "kept_to_initial_face_ratio": float(kept_to_initial_face_ratio),
            "largest_kept_component_share": float(largest_kept_component_share),
            "object_isolation_status": "success" if final_faces > 0 else "failed",
            "object_isolation_method": isolation_method,
            "isolation_confidence": float(best_scores["total_score"]),
            "selected_component_score": float(best_scores["total_score"]),
            "compactness_score": float(best_scores.get("compactness_score", 0.0)),
            "used_sam2": used_sam2,
            "mask_support_ratio": float(best_scores.get("mask_score", 0.0)),
            "point_cloud_support_ratio": float(best_scores.get("pc_score", 0.0)),
            "supported_view_count": int(mask_supports.get(ranked[primary_idx_in_ranked][1], {}).get("supported_view_count", 0)) if ranked else 0,
            "reason_if_geometric_fallback": reason_if_geometric_fallback,
            "primary_assignment_result": primary_assignment_result,
        }

        # Debug export
        if output_dir:
            try:
                final_mesh.export(str(output_dir / "debug_isolated_mesh.obj"))
            except Exception: pass

        return final_mesh, stats

    def _remove_chromatic_leakage_tracked(
        self,
        working_geom: trimesh.Trimesh,
        original_mesh: trimesh.Trimesh,
        background_rgb: Optional[Tuple[int, int, int]] = None,
        background_tolerance: int = 35,
    ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        """
        Removes components dominated by background-colored vertex colors.

        - If `background_rgb` is provided (from ColorProfiler), filter by L∞ distance
          to that color (generic, product-agnostic).
        - Otherwise, fall back to the legacy hardcoded orange filter for backwards-compat
          with existing restaurant-tray captures.
        """
        stats = {
            "chromatic_removed_faces": 0,
            "chromatic_leakage_detected": False,
            "chromatic_filter_mode": "background_rgb" if background_rgb is not None else "legacy_orange",
            "background_rgb": list(background_rgb) if background_rgb is not None else None,
        }

        if not hasattr(original_mesh.visual, 'vertex_colors') or original_mesh.visual.vertex_colors is None:
            return working_geom, stats

        if 'face_indices' not in working_geom.metadata:
            return working_geom, stats

        vertex_colors = original_mesh.visual.vertex_colors

        components = working_geom.split(only_watertight=False)
        if not components:
            return working_geom, stats

        kept_components = []
        for comp in components:
            if 'face_indices' not in comp.metadata:
                kept_components.append(comp)
                continue

            comp_face_indices = comp.metadata['face_indices']
            original_faces = original_mesh.faces[comp_face_indices]
            original_verts_indices = np.unique(original_faces)
            comp_colors = vertex_colors[original_verts_indices][:, :3]

            if background_rgb is not None:
                # Generic background-distance filter
                bg = np.asarray(background_rgb, dtype=np.int16).reshape(1, 3)
                diff = np.abs(comp_colors.astype(np.int16) - bg)
                is_bg = np.max(diff, axis=1) <= background_tolerance
                bg_ratio = float(np.count_nonzero(is_bg) / max(len(comp_colors), 1))
                ratio_label = f"bg_ratio={bg_ratio:.2%}"
            else:
                # Legacy orange filter (backwards-compat for restaurant-tray captures)
                R, G, B = comp_colors[:, 0], comp_colors[:, 1], comp_colors[:, 2]
                is_bg = (R > 140) & (G > 70) & (B < 110) & (R > G + 25)
                bg_ratio = float(np.count_nonzero(is_bg) / max(len(comp_colors), 1))
                ratio_label = f"orange_ratio={bg_ratio:.2%}"

            should_prune = False
            if len(comp.faces) < 500 and bg_ratio > 0.40:
                should_prune = True
            elif bg_ratio > 0.75:
                should_prune = True

            if should_prune:
                stats["chromatic_removed_faces"] += len(comp.faces)
                stats["chromatic_leakage_detected"] = True
                logger.info(
                    f"Pruning chromatic leakage component [{stats['chromatic_filter_mode']}]: "
                    f"faces={len(comp.faces)}, {ratio_label}"
                )
            else:
                kept_components.append(comp)

        if not kept_components:
            return working_geom, stats  # Safety: don't return empty

        final_mesh = trimesh.util.concatenate(kept_components) if len(kept_components) > 1 else kept_components[0]
        all_indices = []
        for c in kept_components:
            if 'face_indices' in c.metadata:
                all_indices.extend(c.metadata['face_indices'])
        final_mesh.metadata['face_indices'] = np.array(all_indices)

        return final_mesh, stats
