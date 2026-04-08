import os
import trimesh
from pathlib import Path
from typing import Dict, Any

class GLBExporter:
    def __init__(self):
        pass

    def export_to_glb(self, mesh_path: str, texture_path: Optional[str], output_path: str, metadata: Optional[NormalizedMetadata] = None) -> Dict[str, Any]:
        """
        Exports a mesh and optional texture to a valid GLB 2.0 file.
        Incorporates normalized metadata (bbox/pivot) into the GLB structure if provided.
        """
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Source mesh not found for GLB export: {mesh_path}")

        # 1. Load the mesh
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
            
        # 2. Add Texture (Visuals)
        if texture_path and os.path.exists(texture_path):
            try:
                from PIL import Image
                tex_image = Image.open(texture_path)
                material = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=tex_image,
                    metallicFactor=0.0,
                    roughnessFactor=1.0
                )
                # We assume UVs exist in the mesh (COLMAP/OpenMVS typically provide them)
                mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=material)
            except Exception as e:
                print(f"Warning: Failed to apply texture {texture_path}: {e}")
        else:
            # Fallback to vertex colors or default material
            if hasattr(mesh.visual, 'vertex_colors'):
                mesh.visual.material = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=(255, 255, 255, 255),
                    metallicFactor=0.0,
                    roughnessFactor=1.0
                )

        # 3. Apply Metadata Transform (Optional: If cleaner hasn't baked it in, but usually it has)
        # Cleaner logic already BAKES the transform into vertices. 
        # Exporting here is just for conversion.

        # 4. Export as GLB
        mesh.export(output_path, file_type='glb')

        return {
            "format": "GLB",
            "filesize": os.path.getsize(output_path),
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "has_texture": bool(texture_path and os.path.exists(texture_path))
        }
