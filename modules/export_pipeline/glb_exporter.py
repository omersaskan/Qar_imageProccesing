import os
from typing import Dict, Any, Optional

import trimesh
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata


class GLBExporter:
    def __init__(self):
        pass

    def _load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        return mesh

    def _inspect_visuals(self, mesh: trimesh.Trimesh) -> Dict[str, bool]:
        has_uv = False
        has_material = False

        try:
            has_uv = (
                hasattr(mesh.visual, "uv")
                and mesh.visual.uv is not None
                and len(mesh.visual.uv) > 0
            )
        except Exception:
            has_uv = False

        try:
            has_material = (
                hasattr(mesh.visual, "material")
                and mesh.visual.material is not None
            )
        except Exception:
            has_material = False

        return {
            "has_uv": bool(has_uv),
            "has_material": bool(has_material),
        }

    def export(
        self,
        mesh_path: str,
        output_path: str,
        profile_name: str = "standard",
        texture_path: Optional[str] = None,
        metadata: Optional[NormalizedMetadata] = None,
    ) -> Dict[str, Any]:
        """
        Worker-compatible wrapper.
        """
        return self.export_to_glb(
            mesh_path=mesh_path,
            texture_path=texture_path,
            output_path=output_path,
            metadata=metadata,
        )

    def export_to_glb(
        self,
        mesh_path: str,
        texture_path: Optional[str],
        output_path: str,
        metadata: Optional[NormalizedMetadata] = None,
    ) -> Dict[str, Any]:
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Source mesh not found for GLB export: {mesh_path}")

        mesh = self._load_mesh(mesh_path)
        visual_info = self._inspect_visuals(mesh)

        used_texture_path = None
        texture_applied_successfully = False
        texture_warning = None

        if texture_path and os.path.exists(texture_path):
            used_texture_path = texture_path

            if visual_info["has_uv"]:
                try:
                    from PIL import Image

                    with Image.open(texture_path) as texture_image:
                        tex_image = texture_image.convert("RGBA").copy()
                    material = trimesh.visual.material.PBRMaterial(
                        baseColorTexture=tex_image,
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    )

                    # apply texture only if UV exists
                    mesh.visual = trimesh.visual.TextureVisuals(
                        uv=mesh.visual.uv,
                        material=material,
                    )

                    texture_applied_successfully = True
                    visual_info = self._inspect_visuals(mesh)

                except Exception as e:
                    texture_applied_successfully = False
                    texture_warning = f"Texture apply failed: {e}"
            else:
                texture_applied_successfully = False
                texture_warning = "Texture file exists but mesh has no UV coordinates"
        else:
            if texture_path:
                texture_warning = f"Texture path missing on disk: {texture_path}"

        # fallback material if no texture successfully applied
        if not texture_applied_successfully:
            try:
                if hasattr(mesh.visual, "vertex_colors"):
                    mesh.visual.material = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=(255, 255, 255, 255),
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    )
            except Exception:
                pass

        # metadata transform is expected to be already baked by cleaner/alignment
        glb_bytes = mesh.export(file_type="glb")
        with open(output_path, "wb") as glb_file:
            glb_file.write(glb_bytes)

        result = {
            "format": "GLB",
            "filesize": os.path.getsize(output_path),
            "vertex_count": int(len(mesh.vertices)) if hasattr(mesh, "vertices") else 0,
            "face_count": int(len(mesh.faces)) if hasattr(mesh, "faces") else 0,
            "has_uv": visual_info["has_uv"],
            "has_material": visual_info["has_material"],
            "used_texture_path": used_texture_path,
            "texture_applied_successfully": texture_applied_successfully,
        }

        if texture_warning:
            result["texture_warning"] = texture_warning

        if metadata is not None:
            result["bbox_min"] = metadata.bbox_min
            result["bbox_max"] = metadata.bbox_max
            result["pivot_offset"] = metadata.pivot_offset
            result["final_polycount"] = metadata.final_polycount

        return result
