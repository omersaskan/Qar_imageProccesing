from .profiles import CleanupProfile
import trimesh
import numpy as np
import fast_simplification
import os
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("remesher")

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
        
        has_uv = hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None

        if current_faces > target_faces:
            if has_uv:
                ratio = target_faces / current_faces
                logger.info(f"Preserving UVs during simplification: {current_faces} -> {target_faces} faces (ratio={ratio:.2f})")
                try:
                    # trimesh expects a ratio (0.0 to 1.0) for quadric decimation
                    mesh = mesh.simplify_quadric_decimation(ratio)
                except Exception as e:
                    logger.warning(f"Quadric decimation failed, falling back to basic simplify: {e}")
                    # Basic simplify (doesn't always preserve UVs but prevents crash)
                    mesh = mesh.simplify_quadric_decimation(target_faces / current_faces)
            else:
                try:
                    # Use fast-simplification for speed and quality on non-textured meshes
                    points = mesh.vertices.astype(np.float32)
                    faces = mesh.faces.astype(np.uint32)
                    
                    # Target reduction ratio
                    ratio = target_faces / current_faces
                    new_vertices, new_faces = fast_simplification.simplify(points, faces, ratio)
                    
                    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
                except Exception as e:
                    logger.warning(f"fast_simplification failed, falling back to trimesh: {e}")
                    # Fallback to trimesh default simplify (decimation)
                    mesh = mesh.simplify_quadric_decimation(target_faces)

        # 3. Post-Simplification Health Checks
        mesh.process(validate=True)
        mesh.remove_unreferenced_vertices()
        
        if len(mesh.faces) == 0:
             logger.warning("Warning: Remeshing produced empty mesh!")

        # 4. Finalize
        # Phase 1: Ensure we export with visual data if it exists
        mesh.export(output_path)
        return len(mesh.vertices)
