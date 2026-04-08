import trimesh
import numpy as np
from typing import Tuple, List, Optional
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("isolation")

class MeshIsolator:
    """
    Isolates the main product from a scene mesh by removing planes and disconnected noise.
    """
    def __init__(self):
        pass

    def isolate_product(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, dict]:
        """
        Main entry point for product isolation.
        Returns (isolated_mesh, stats).
        """
        if len(mesh.vertices) == 0:
            return mesh, {"error": "Empty mesh"}
        stats = {
            "initial_faces": len(mesh.faces),
            "initial_vertices": len(mesh.vertices),
            "removed_planes": 0,
            "removed_islands": 0,
            "removed_plane_faces": 0,
            "removed_plane_vertices": 0
        }

        # 1. Remove dominant planes (Table/Floor) - Robust Iterative
        current_mesh = mesh.copy()
        
        for i in range(3): # Increased iterations for complex scenes
            try:
                # Use a larger sample for plane fitting in messy scenes
                if len(current_mesh.vertices) < 500: break
                
                # Sample points for robust RANSAC-like plane search (simulated here with better fit)
                pts = current_mesh.vertices[np.random.choice(len(current_mesh.vertices), min(2000, len(current_mesh.vertices)), replace=False)]
                plane_origin, plane_normal = trimesh.points.plane_fit(pts)
                
                # Filter points close to the plane
                distances = np.abs(np.dot(current_mesh.vertices - plane_origin, plane_normal))
                
                # Adaptive threshold based on mesh scale, but capped to prevent removing products
                bbox_diag = np.linalg.norm(current_mesh.bounds[1] - current_mesh.bounds[0])
                # Product-safe threshold: 0.5% of diagonal or 1cm, whichever is smaller
                threshold = min(0.01, bbox_diag * 0.005) 
                
                inliers = np.where(distances < threshold)[0]
                
                if len(inliers) > len(current_mesh.vertices) * 0.05: # >5% check
                    # Track faces to be removed
                    face_mask = np.any(np.isin(current_mesh.faces, inliers), axis=1)
                    stats["removed_plane_faces"] += int(np.sum(face_mask))
                    
                    # Remove vertices
                    vertex_mask = np.ones(len(current_mesh.vertices), dtype=bool)
                    vertex_mask[inliers] = False
                    stats["removed_plane_vertices"] += len(inliers)
                    current_mesh.update_vertices(vertex_mask)
                    stats["removed_planes"] += 1
                else:
                    break
            except Exception as e:
                logger.warning(f"Plane detection attempt {i} failed: {e}")
                break

        # 2. Split into connected components
        components = current_mesh.split(only_watertight=False)
        if not components:
            return current_mesh, stats

        logger.info(f"Found {len(components)} connected components.")
        
        # 3. Score components to find the product
        # Phase 3: Enhanced Scoring (Face Count, Centrality, Flatness Penalty, Compactness)
        best_comp = None
        best_score = -1.0
        
        for comp in components:
            f_count = len(comp.faces)
            if f_count < 200: # Increased noise threshold
                stats["removed_islands"] += 1
                continue
                
            # A. Face Score
            face_score = min(1.0, f_count / 10000.0)
            
            # B. Centrality (Relative to original mesh center/centroid)
            center_dist = np.linalg.norm(comp.centroid)
            centrality_score = np.exp(-center_dist / 0.5) # Decay faster
            
            # C. Compactness (Volume/BBoxVolume)
            bbox = comp.bounds[1] - comp.bounds[0]
            bbox_vol = np.prod(bbox)
            # Use convex hull volume for non-watertight meshes
            compactness = comp.convex_hull.volume / bbox_vol if bbox_vol > 0 else 0
            
            # D. Flatness Penalty (Is it a forgotten plane segment or a slab?)
            # bbox_aspect = min_dim / max_dim
            min_dim = np.min(bbox)
            max_dim = np.max(bbox)
            aspect_ratio = min_dim / max_dim if max_dim > 0 else 0
            flatness_penalty = 1.0 if aspect_ratio > 0.1 else (aspect_ratio * 10.0)
            
            # Weighted Score
            score = (
                face_score * 0.3 +
                centrality_score * 0.3 +
                compactness * 0.2 +
                flatness_penalty * 0.2
            )
            
            if score > best_score:
                best_score = score
                best_comp = comp

        if best_comp is None:
            logger.warning("No suitable product component found! Returning largest component as fallback.")
            best_comp = max(components, key=lambda c: len(c.faces))

        stats.update({
            "final_faces": len(best_comp.faces),
            "final_vertices": len(best_comp.vertices),
            "component_count": len(components),
            "island_count": len(components) - 1,
            "selected_component_score": float(best_score),
            "compactness_score": float(compactness if 'compactness' in locals() else 0),
            "flatness_score": float(aspect_ratio if 'aspect_ratio' in locals() else 0)
        })
        
        # Calculate Ratios
        stats["removed_plane_face_share"] = stats["removed_plane_faces"] / stats["initial_faces"] if stats["initial_faces"] > 0 else 0
        stats["removed_plane_vertex_ratio"] = stats["removed_plane_vertices"] / stats["initial_vertices"] if stats["initial_vertices"] > 0 else 0

        return best_comp, stats
