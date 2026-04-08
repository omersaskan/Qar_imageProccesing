import trimesh
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("mesh_selector")

class MeshSelector:
    """
    Analyzes and scores reconstruction mesh candidates to identify the best 'product' mesh.
    Useful for choosing between Poisson, Delaunay, or multiple connected components.
    """
    def __init__(self):
        pass

    def score_mesh(self, mesh: trimesh.Trimesh) -> Dict[str, float]:
        """
        Calculates scores for a mesh based on product-centric criteria.
        """
        if len(mesh.vertices) == 0:
            return {"total_score": 0.0}

        # 1. Face Count (Prefer high detail but reasonably bounded)
        face_count = len(mesh.faces)
        # Log scale scoring for face count
        face_score = min(1.0, np.log10(face_count + 1) / 5.0) 

        # 2. Compactness (Is it a solid object or sparse islands?)
        # Volume / BBox Volume
        bbox_size = mesh.bounds[1] - mesh.bounds[0]
        bbox_volume = np.prod(bbox_size)
        mesh_volume = abs(mesh.volume) if mesh.is_watertight else mesh.convex_hull.volume
        compactness = mesh_volume / bbox_volume if bbox_volume > 0 else 0
        
        # 3. Centrality (Is it near the origin?)
        center = mesh.centroid
        center_dist = np.linalg.norm(center)
        # Normalized distance: higher score for closer to origin
        centrality_score = np.exp(-center_dist / 10.0)

        # 4. Flatness Penalty (Is it a table/wall?)
        # Use PCA to check if one dimension is very small
        eigenvalues = np.sort(mesh.principal_inertia_components)
        if eigenvalues[2] > 0:
            flatness = eigenvalues[0] / eigenvalues[2]
        else:
            flatness = 0
            
        # Product flatness: Most products have some depth. 
        # Tables/Walls have extremely low depth ratio.
        flatness_penalty = 1.0 if flatness > 0.05 else (flatness * 20.0)

        # 5. Connected Components Score
        components = mesh.split(only_watertight=False)
        comp_count = len(components)
        # Extreme fragmentation is a bad sign for a 'product'
        fragment_penalty = np.exp(-(comp_count - 1) / 5.0)

        total_score = (
            face_score * 0.2 +
            compactness * 0.2 +
            centrality_score * 0.2 +
            flatness_penalty * 0.3 + # Increased weight for product-likeness
            fragment_penalty * 0.1
        )

        return {
            "total_score": total_score,
            "face_score": face_score,
            "compactness": compactness,
            "centrality_score": centrality_score,
            "flatness_penalty": flatness_penalty,
            "fragment_penalty": fragment_penalty,
            "face_count": face_count,
            "component_count": comp_count
        }

    def select_best_mesh(self, mesh_paths: List[str]) -> Optional[str]:
        """
        Takes a list of mesh paths and returns the path with the highest score.
        """
        best_path = None
        best_score = -1.0

        for path_str in mesh_paths:
            path = Path(path_str)
            if not path.exists():
                continue

            try:
                mesh = trimesh.load(str(path))
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)
                
                scores = self.score_mesh(mesh)
                logger.info(f"Mesh {path.name} score: {scores['total_score']:.4f} (Faces: {scores['face_count']})")
                
                if scores["total_score"] > best_score:
                    best_score = scores["total_score"]
                    best_path = path_str
            except Exception as e:
                logger.error(f"Failed to score mesh {path.name}: {e}")

        return best_path
