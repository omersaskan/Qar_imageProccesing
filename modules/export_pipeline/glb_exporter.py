import os
from typing import Dict, Any, Optional
import trimesh
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata


class GLBExporter:
    def __init__(self):
        pass

    def _inspect_visuals(self, mesh: trimesh.Trimesh) -> Dict[str, bool]:
        has_uv = False
        has_material = False
        try:
            has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
        except Exception:
            has_uv = False

        try:
            has_material = hasattr(mesh.visual, "material") and mesh.visual.material is not None
        except Exception:
            has_material = False

        return {"has_uv": bool(has_uv), "has_material": bool(has_material)}

    def export(
        self,
        mesh_path: str,
        output_path: str,
        profile_name: str = "standard",
        texture_path: Optional[str] = None,
        metadata: Optional[NormalizedMetadata] = None,
    ) -> Dict[str, Any]:
        return self.export_to_glb(mesh_path, texture_path, output_path, metadata)

    def export_to_glb(
        self,
        mesh_path: str,
        texture_path: Optional[str],
        output_path: str,
        metadata: Optional[NormalizedMetadata] = None,
    ) -> Dict[str, Any]:
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Source mesh not found for GLB export: {mesh_path}")

        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        visual_info = self._inspect_visuals(mesh)
        used_texture_path = None
        texture_applied_successfully = False

        if texture_path and os.path.exists(texture_path):
            used_texture_path = texture_path
            if visual_info["has_uv"]:
                try:
                    from PIL import Image

                    tex_image = Image.open(texture_path)
                    material = trimesh.visual.material.PBRMaterial(
                        baseColorTexture=tex_image,
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    )
                    mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, material=material)
                    texture_applied_successfully = True
                    visual_info = self._inspect_visuals(mesh)
                except Exception:
                    texture_applied_successfully = False
            else:
                # texture exists but mesh has no UV
                texture_applied_successfully = False
        else:
            if hasattr(mesh.visual, "vertex_colors"):
                try:
                    mesh.visual.material = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=(255, 255, 255, 255),
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    )
                except Exception:
                    pass

        mesh.export(output_path, file_type="glb")

        return {
            "format": "GLB",
            "filesize": os.path.getsize(output_path),
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "has_uv": visual_info["has_uv"],
            "has_material": visual_info["has_material"],
            "used_texture_path": used_texture_path,
            "texture_applied_successfully": texture_applied_successfully,
        }