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
            "removed_islands": 0
        }

        # 1. Remove dominant planes (Table/Floor)
        current_mesh = mesh.copy()
        stats["removed_plane_faces"] = 0
        
        for i in range(2):
            try:
                # Find plane using vertices
                plane_origin, plane_normal = trimesh.points.plane_fit(current_mesh.vertices)
                # Filter points close to the plane
                distances = np.abs(np.dot(current_mesh.vertices - plane_origin, plane_normal))
                inliers = np.where(distances < 0.05)[0] # 5cm threshold for plane
                
                if len(inliers) > len(current_mesh.vertices) * 0.1: # If plane contains >10% of vertices
                    logger.info(f"Removing dominant plane with {len(inliers)} vertices.")
                    
                    # Track faces to be removed for stats
                    # A face is removed if any of its vertices are in the plane
                    face_mask = np.any(np.isin(current_mesh.faces, inliers), axis=1)
                    stats["removed_plane_faces"] += int(np.sum(face_mask))
                    
                    # Effective removal
                    vertex_mask = np.ones(len(current_mesh.vertices), dtype=bool)
                    vertex_mask[inliers] = False
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
        # Criteria: Centrality, Face Count, Non-flatness
        best_comp = None
        best_score = -1.0
        
        for comp in components:
            f_count = len(comp.faces)
            if f_count < 100: # Ignore tiny noise
                stats["removed_islands"] += 1
                continue
                
            # Score
            center_dist = np.linalg.norm(comp.centroid)
            occupancy = comp.area # Surface area as a proxy for size
            
            # Heuristic score: higher is better
            # Prefers central and substantial objects
            score = (f_count * 0.5) * np.exp(-center_dist / 5.0)
            
            # Penalty for being too flat (planes)
            bbox = comp.bounds[1] - comp.bounds[0]
            if np.min(bbox) < 0.02 or np.min(bbox) / np.max(bbox) < 0.1:
                score *= 0.1 # Severe penalty for thin slabs
            
            if score > best_score:
                best_score = score
                best_comp = comp

        if best_comp is None:
            logger.warning("No suitable product component found! Returning largest component as fallback.")
            best_comp = max(components, key=lambda c: len(c.faces))

        stats["final_faces"] = len(best_comp.faces)
        stats["final_vertices"] = len(best_comp.vertices)
        stats["island_count"] = len(components) - 1
        
        return best_comp, stats
