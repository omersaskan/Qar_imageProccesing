from typing import Dict, List, Tuple

import trimesh
from typing import Dict, Tuple

class BBoxExtractor:
    def __init__(self):
        pass

    def extract(self, mesh_path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Extracts the real axis-aligned bounding box from the mesh.
        """
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
            
        bounds = mesh.bounds
        min_corner = {"x": float(bounds[0][0]), "y": float(bounds[0][1]), "z": float(bounds[0][2])}
        max_corner = {"x": float(bounds[1][0]), "y": float(bounds[1][1]), "z": float(bounds[1][2])}

        return min_corner, max_corner
