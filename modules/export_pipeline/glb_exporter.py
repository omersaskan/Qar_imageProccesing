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

    def _material_has_texture(self, mesh: trimesh.Trimesh, slot_name: str = "baseColorTexture") -> bool:
        try:
            material = getattr(mesh.visual, "material", None)
            if material is None:
                return False
            
            tex = getattr(material, slot_name, None)
            # Legacy fallback for older trimesh versions or generic simple materials
            if tex is None and slot_name == "baseColorTexture":
                 tex = getattr(material, "image", None)
                
            if tex is not None:
                # PIL Image or similar
                if hasattr(tex, "size"):
                    if tex.size[0] <= 2 and tex.size[1] <= 2:
                        return False
                    return True
                # Numpy array
                if hasattr(tex, "shape"):
                    if tex.shape[0] <= 2 and tex.shape[1] <= 2:
                        return False
                    return True
                # Fallback if it has something but we can't tell its size easily
                return True
        except Exception:
            return False
        return False

    def _flatten_loaded_asset(self, loaded) -> tuple[trimesh.Trimesh, list[trimesh.Trimesh]]:
        if isinstance(loaded, trimesh.Scene):
            meshes = [
                geom
                for geom in loaded.geometry.values()
                if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0
            ]
            if not meshes:
                raise ValueError("Exported GLB contains no renderable mesh geometry.")
            combined = trimesh.util.concatenate([mesh.copy() for mesh in meshes])
            return combined, meshes

        if isinstance(loaded, trimesh.Trimesh):
            return loaded, [loaded]

        raise ValueError("Exported asset is not a mesh or scene.")

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
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            raise ValueError(f"Source asset is not a polygon mesh: {mesh_path}")
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError(f"Source mesh has no renderable geometry: {mesh_path}")
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

    def inspect_exported_asset(self, glb_path: str) -> Dict[str, Any]:
        if not os.path.exists(glb_path):
            raise FileNotFoundError(f"Exported GLB not found: {glb_path}")
        if os.path.getsize(glb_path) <= 0:
            raise ValueError(f"Exported GLB is empty: {glb_path}")

        loaded = trimesh.load(glb_path, force="scene")
        combined, meshes = self._flatten_loaded_asset(loaded)

        if len(combined.vertices) == 0 or len(combined.faces) == 0:
            raise ValueError(f"Exported GLB has no renderable geometry: {glb_path}")

        has_uv = False
        has_material = False
        has_embedded_texture = False
        texture_count = 0
        material_count = 0
        
        # semantics
        basecolor_present = False
        metallic_roughness_present = False
        normal_present = False
        occlusion_present = False
        emissive_present = False

        for mesh in meshes:
            vis = self._inspect_visuals(mesh)
            if vis["has_uv"]:
                has_uv = True
                
            if getattr(mesh.visual, "material", None) is not None:
                has_material = True
                material_count += 1
                
                # Check specific PBR slots
                if self._material_has_texture(mesh, "baseColorTexture"):
                    basecolor_present = True
                    has_embedded_texture = True
                    texture_count += 1
                if self._material_has_texture(mesh, "metallicRoughnessTexture"):
                    metallic_roughness_present = True
                    texture_count += 1
                if self._material_has_texture(mesh, "normalTexture"):
                    normal_present = True
                    texture_count += 1
                if self._material_has_texture(mesh, "occlusionTexture"):
                    occlusion_present = True
                    texture_count += 1
                if self._material_has_texture(mesh, "emissiveTexture"):
                    emissive_present = True
                    texture_count += 1
                    
        component_count = len(combined.split(only_watertight=False))
        bounds = combined.bounds
        
        # Honest Integrity status (preservation)
        if has_embedded_texture and has_uv and has_material:
            integrity_status = "complete"
        elif has_embedded_texture or has_uv:
            integrity_status = "degraded"
        else:
            integrity_status = "missing"
            
        # Honest Semantic status (richness)
        if not has_uv:
            semantic_status = "geometry_only"
        elif not has_embedded_texture:
            semantic_status = "uv_only"
        elif basecolor_present and normal_present and metallic_roughness_present:
            semantic_status = "pbr_complete"
        elif basecolor_present and (normal_present or metallic_roughness_present):
            semantic_status = "pbr_partial"
        elif basecolor_present:
            semantic_status = "diffuse_textured"
        else:
            semantic_status = "material_incomplete"

        return {
            "vertex_count": int(len(combined.vertices)),
            "face_count": int(len(combined.faces)),
            "geometry_count": int(len(meshes)),
            "component_count": int(component_count),
            "has_uv": bool(has_uv),
            "has_material": bool(has_material),
            "has_embedded_texture": bool(has_embedded_texture),
            "texture_count": texture_count,
            "material_count": material_count,
            "texture_integrity_status": integrity_status,
            "material_semantic_status": semantic_status,
            "basecolor_present": basecolor_present,
            "metallic_roughness_present": metallic_roughness_present,
            "normal_present": normal_present,
            "occlusion_present": occlusion_present,
            "emissive_present": emissive_present,
            "material_integrity_status": "present" if has_material else "missing",
            "bounds_min": {
                "x": float(bounds[0][0]),
                "y": float(bounds[0][1]),
                "z": float(bounds[0][2]),
            },
            "bounds_max": {
                "x": float(bounds[1][0]),
                "y": float(bounds[1][1]),
                "z": float(bounds[1][2]),
            },
            "bbox": {
                "x": float(bounds[1][0] - bounds[0][0]),
                "y": float(bounds[1][1] - bounds[0][1]),
                "z": float(bounds[1][2] - bounds[0][2]),
            },
            "ground_offset": abs(float(bounds[0][2])),
        }
