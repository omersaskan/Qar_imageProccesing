import os
import trimesh
from pathlib import Path
from typing import Dict, Any

class GLBExporter:
    def __init__(self):
        pass

    def export(self, mesh_path: str, output_path: str, profile_name: str) -> Dict[str, Any]:
        """
        Real GLB generation using trimesh.
        Loads the provided mesh and exports it as a valid GLB 2.0 file.
        """
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Source mesh not found for GLB export: {mesh_path}")

        # 1. Load the mesh (OBJ/PLY/etc)
        scene_or_mesh = trimesh.load(mesh_path)
        
        # 2. Export as GLB
        scene_or_mesh.export(output_path, file_type='glb')

        # 3. Collect Metadata
        is_point_cloud = isinstance(scene_or_mesh, trimesh.PointCloud)
        vertex_count = len(scene_or_mesh.vertices) if hasattr(scene_or_mesh, 'vertices') else 0
        face_count = len(scene_or_mesh.faces) if hasattr(scene_or_mesh, 'faces') else 0

        return {
            "format": "GLB",
            "profile": profile_name,
            "stub": False,
            "is_point_cloud": is_point_cloud,
            "filesize": os.path.getsize(output_path),
            "vertex_count": vertex_count,
            "face_count": face_count
        }
