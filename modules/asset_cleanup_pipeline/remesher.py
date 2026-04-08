import os
from pathlib import Path
from .profiles import CleanupProfile

import trimesh
import fast_simplification
from .profiles import CleanupProfile

class Remesher:
    def __init__(self):
        pass

    def process(self, input_path: str, output_path: str, profile: CleanupProfile) -> int:
        """
        Performs real mesh simplification and repair.
        Returns the final vertex count.
        """
        mesh = trimesh.load(input_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        if len(mesh.faces) == 0:
            return 0

        # 1. Mesh Repair
        # Fix normals, remove degenerate faces, fill small holes
        mesh.fill_holes()
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()

        # 2. Simplification
        # Calculate target faces
        target_faces = profile.target_polycount
        current_faces = len(mesh.faces)

        if current_faces > target_faces:
            # Use fast-simplification for speed and quality
            points = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.uint32)
            
            # Target reduction ratio
            ratio = target_faces / current_faces
            new_vertices, new_faces = fast_simplification.simplify(points, faces, ratio)
            
            mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

        # 3. Finalize
        mesh.export(output_path)
        return len(mesh.vertices)
