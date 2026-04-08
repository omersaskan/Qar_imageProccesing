from .profiles import CleanupProfile
import trimesh
import numpy as np
import fast_simplification

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
        mesh.process(validate=True)

        # 2. Simplification
        # Calculate target faces
        target_faces = profile.target_polycount
        current_faces = len(mesh.faces)

        if current_faces > target_faces:
            try:
                # Use fast-simplification for speed and quality
                points = mesh.vertices.astype(np.float32)
                faces = mesh.faces.astype(np.uint32)
                
                # Target reduction ratio
                ratio = target_faces / current_faces
                new_vertices, new_faces = fast_simplification.simplify(points, faces, ratio)
                
                mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
            except Exception as e:
                print(f"Warning: fast_simplification failed, falling back to trimesh: {e}")
                # Fallback to trimesh default simplify (decimation)
                mesh = mesh.simplify_quadric_decimation(target_faces)

        # 3. Post-Simplification Health Checks
        mesh.process(validate=True)
        mesh.remove_unreferenced_vertices()
        
        if len(mesh.faces) == 0:
             print("Warning: Remeshing produced empty mesh!")

        # 4. Finalize
        mesh.export(output_path)
        return len(mesh.vertices)
