import os
import struct
import json
from typing import Dict, Any, Optional
import trimesh
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata


def inspect_glb_primitive_attributes(glb_path: str) -> Dict[str, Any]:
    """
    STRICT GLB inspector that reads the raw JSON chunk to verify accessors.
    This avoids trimesh false positives where normals are computed on the fly.
    """
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB file not found for inspection: {glb_path}")

    with open(glb_path, "rb") as f:
        # Header (12 bytes): magic (4), version (4), length (4)
        magic = f.read(4)
        if magic != b"glTF":
            # Fallback for plain .gltf files (JSON)
            f.seek(0)
            try:
                gltf = json.load(f)
            except Exception:
                raise ValueError(f"Not a valid glTF/GLB file: {glb_path}")
        else:
            _version = struct.unpack("<I", f.read(4))[0]
            _length = struct.unpack("<I", f.read(4))[0]

            # First chunk must be JSON
            chunk_length = struct.unpack("<I", f.read(4))[0]
            chunk_type = f.read(4)
            if chunk_type != b"JSON":
                # Some GLBs might have padding or different order, but spec says JSON is first.
                # We'll try to find the JSON chunk.
                found_json = False
                f.seek(12)
                while True:
                    c_len_raw = f.read(4)
                    if not c_len_raw: break
                    c_len = struct.unpack("<I", c_len_raw)[0]
                    c_type = f.read(4)
                    if c_type == b"JSON":
                        json_data = f.read(c_len).decode("utf-8")
                        gltf = json.loads(json_data)
                        found_json = True
                        break
                    else:
                        f.seek(c_len, 1)
                if not found_json:
                    raise ValueError("Could not find JSON chunk in GLB")
            else:
                json_data = f.read(chunk_length).decode("utf-8")
                gltf = json.loads(json_data)

    all_primitives_have_position = True
    all_primitives_have_normal = True
    all_textured_primitives_have_texcoord_0 = True
    primitive_reports = []

    meshes = gltf.get("meshes", [])
    materials = gltf.get("materials", [])

    for mesh_idx, mesh in enumerate(meshes):
        for prim_idx, prim in enumerate(mesh.get("primitives", [])):
            attrs = prim.get("attributes", {})
            has_pos = "POSITION" in attrs
            has_norm = "NORMAL" in attrs
            has_uv = "TEXCOORD_0" in attrs

            # Detect if primitive is textured by checking its material
            is_textured = False
            mat_idx = prim.get("material")
            if mat_idx is not None and isinstance(mat_idx, int) and mat_idx < len(materials):
                mat = materials[mat_idx]
                pbr = mat.get("pbrMetallicRoughness", {})
                if (
                    "baseColorTexture" in pbr
                    or "metallicRoughnessTexture" in pbr
                    or "normalTexture" in mat
                    or "occlusionTexture" in mat
                    or "emissiveTexture" in mat
                ):
                    is_textured = True

            missing = []
            if not has_pos:
                missing.append("POSITION")
                all_primitives_have_position = False
            if not has_norm:
                missing.append("NORMAL")
                all_primitives_have_normal = False
            if is_textured and not has_uv:
                missing.append("TEXCOORD_0")
                all_textured_primitives_have_texcoord_0 = False

            report = {
                "mesh_index": mesh_idx,
                "primitive_index": prim_idx,
                "attributes": list(attrs.keys()),
                "has_position": has_pos,
                "has_normal": has_norm,
                "has_texcoord_0": has_uv,
                "is_textured": is_textured,
                "missing_attributes": missing,
            }
            primitive_reports.append(report)

    if not primitive_reports:
        all_primitives_have_position = False
        all_primitives_have_normal = False

    return {
        "all_primitives_have_position": all_primitives_have_position,
        "all_primitives_have_normal": all_primitives_have_normal,
        "all_textured_primitives_have_texcoord_0": all_textured_primitives_have_texcoord_0,
        "primitive_attribute_report": primitive_reports,
        # Legacy mappings for backward compatibility during transition
        "has_position_accessor": all_primitives_have_position,
        "has_normal_accessor": all_primitives_have_normal,
        "has_texcoord_0_accessor": all_textured_primitives_have_texcoord_0,
    }


class GLBExporter:
    def __init__(self):
        pass

    def _load_mesh(self, mesh_path: str) -> trimesh.Trimesh | trimesh.Scene:
        mesh = trimesh.load(mesh_path)
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

    def _flatten_loaded_asset(self, loaded) -> tuple[Any, list[trimesh.Trimesh]]:
        if isinstance(loaded, trimesh.Scene):
            meshes = [
                geom
                for geom in loaded.geometry.values()
                if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0
            ]
            if not meshes:
                raise ValueError("Exported GLB contains no renderable mesh geometry.")
            return loaded, meshes

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

        loaded = self._load_mesh(mesh_path)
        
        if isinstance(loaded, trimesh.Scene):
            meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        elif isinstance(loaded, trimesh.Trimesh):
            meshes = [loaded]
        else:
            raise ValueError(f"Source asset is not a polygon mesh: {mesh_path}")

        total_verts = sum(len(m.vertices) for m in meshes if hasattr(m, 'vertices'))
        total_faces = sum(len(m.faces) for m in meshes if hasattr(m, 'faces'))
        
        if total_verts == 0 or total_faces == 0:
            raise ValueError(f"Source mesh has no renderable geometry: {mesh_path}")
            
        visual_info = {"has_uv": False, "has_material": False}
        for m in meshes:
            vis = self._inspect_visuals(m)
            if vis["has_uv"]: visual_info["has_uv"] = True
            if vis["has_material"]: visual_info["has_material"] = True

        import logging
        logger = logging.getLogger("glb_exporter")
        
        used_texture_path = None
        texture_applied_successfully = False
        texture_warning = None

        if texture_path and os.path.exists(texture_path):
            used_texture_path = texture_path
            logger.info("Texture path provided for export: %s", used_texture_path)

            if visual_info["has_uv"]:
                try:
                    from PIL import Image

                    with Image.open(texture_path) as texture_image:
                        tex_image = texture_image.convert("RGBA").copy()
                    
                    # Force a sane PBR material
                    material = trimesh.visual.material.PBRMaterial(
                        baseColorTexture=tex_image,
                        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                        doubleSided=True,
                    )
                    logger.info("Forcing PBRMaterial with baseColorFactor=[1.0, 1.0, 1.0, 1.0]")

                    for m in meshes:
                        if hasattr(m.visual, "uv") and m.visual.uv is not None and len(m.visual.uv) > 0:
                            # SPRINT: Black material prevention
                            # Even if it has a material, if it's black or suspicious, we override it.
                            existing_mat = getattr(m.visual, "material", None)
                            is_suspicious = False
                            if existing_mat:
                                logger.info("Existing material found: %s", type(existing_mat).__name__)
                                # Check for black diffuse/baseColor
                                kd = getattr(existing_mat, "diffuse", None)
                                if kd is not None and all(c <= 0 for c in kd[:3]):
                                    is_suspicious = True
                                bcf = getattr(existing_mat, "baseColorFactor", None)
                                if bcf is not None and all(c <= 0 for c in bcf[:3]):
                                    is_suspicious = True
                                    
                                if is_suspicious:
                                    logger.warning("Suspicious (black) material detected, forcing override.")

                            if not existing_mat or is_suspicious:
                                m.visual = trimesh.visual.TextureVisuals(
                                    uv=m.visual.uv,
                                    material=material,
                                )
                                texture_applied_successfully = True
                            else:
                                # Try to inject texture into existing material
                                try:
                                    if hasattr(existing_mat, "baseColorTexture"):
                                        existing_mat.baseColorTexture = tex_image
                                        # Ensure it's not black
                                        if hasattr(existing_mat, "baseColorFactor"):
                                            existing_mat.baseColorFactor = [1.0, 1.0, 1.0, 1.0]
                                        texture_applied_successfully = True
                                    elif hasattr(existing_mat, "image"):
                                        existing_mat.image = tex_image
                                        texture_applied_successfully = True
                                    else:
                                        # Fallback to forcing PBR
                                        m.visual = trimesh.visual.TextureVisuals(
                                            uv=m.visual.uv,
                                            material=material,
                                        )
                                        texture_applied_successfully = True
                                except Exception as e:
                                    logger.warning("Failed to inject texture into existing material: %s. Forcing PBR fallback.", e)
                                    m.visual = trimesh.visual.TextureVisuals(
                                        uv=m.visual.uv,
                                        material=material,
                                    )
                                    texture_applied_successfully = True

                    visual_info["has_uv"] = True
                    visual_info["has_material"] = True
                except Exception as e:
                    texture_applied_successfully = False
                    texture_warning = f"Texture apply failed: {e}"
            else:
                texture_applied_successfully = False
                texture_warning = "Texture file exists but mesh has no UV coordinates"
        else:
            if texture_path:
                texture_warning = f"Texture path missing on disk: {texture_path}"

        if not texture_applied_successfully:
            try:
                fallback_mat = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=(255, 255, 255, 255),
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                )
                for m in meshes:
                    # Apply fallback only if it currently lacks a material entirely
                    if getattr(m.visual, "material", None) is None:
                        try:
                            m.visual.material = fallback_mat
                        except Exception:
                            # If assigning material directly fails (e.g., ColorVisuals strictness), force PBR
                            m.visual = trimesh.visual.TextureVisuals(
                                uv=getattr(m.visual, "uv", None),
                                material=fallback_mat
                            )
            except Exception:
                pass

        for m in meshes:
            # Task 1: GLB normals fix
            m.fix_normals(multibody=True)
            _ = m.vertex_normals # Materialize normals

        if isinstance(loaded, trimesh.Scene):
            glb_bytes = loaded.export(file_type="glb")
        else:
            glb_bytes = meshes[0].export(file_type="glb")
            
        with open(output_path, "wb") as glb_file:
            glb_file.write(glb_bytes)

        # SPRINT: Reload exported GLB to confirm embedded texture
        inspection_result = self.inspect_exported_asset(output_path)
        actual_texture_success = inspection_result["has_embedded_texture"]

        if not actual_texture_success and texture_path and visual_info["has_uv"]:
            texture_warning = "Texture file and UV existed, but embedded texture failed during export."
        elif not actual_texture_success and texture_path and not visual_info["has_uv"]:
            texture_warning = "Texture file exists but mesh has no UV coordinates (Geometry-only fallback)"

        # Accessor check (POSITION, NORMAL, TEXCOORD_0)
        # Acceptance: Textured GLB primitive attributes içinde NORMAL yoksa export result pass olamaz.
        if actual_texture_success and not inspection_result["all_primitives_have_normal"]:
            logger.error("One or more textured GLB primitives are missing NORMAL accessor! This is a delivery blocker.")
            raise ValueError("Export failed: All textured GLB primitives must have NORMAL accessor.")
            
        if inspection_result["all_primitives_have_position"] and \
           inspection_result["all_primitives_have_normal"] and \
           inspection_result["all_textured_primitives_have_texcoord_0"]:
            logger.info("GLB EXPORT SUCCESS: POSITION + NORMAL + TEXCOORD_0 (for textured) present on all primitives.")

        result = {
            "format": "GLB",
            "filesize": os.path.getsize(output_path),
            "vertex_count": total_verts,
            "face_count": total_faces,
            "has_uv": inspection_result["has_uv"],
            "has_material": inspection_result["has_material"],
            "used_texture_path": used_texture_path,
            "texture_applied_successfully": actual_texture_success,
            "has_embedded_texture": actual_texture_success,
            "material_semantic_status": inspection_result["material_semantic_status"],
            "texture_integrity_status": inspection_result["texture_integrity_status"],
            "all_primitives_have_position": inspection_result["all_primitives_have_position"],
            "all_primitives_have_normal": inspection_result["all_primitives_have_normal"],
            "all_textured_primitives_have_texcoord_0": inspection_result["all_textured_primitives_have_texcoord_0"],
            # Legacy fields for backward compat
            "has_position_accessor": inspection_result["all_primitives_have_position"],
            "has_normal_accessor": inspection_result["all_primitives_have_normal"],
            "has_texcoord_0_accessor": inspection_result["all_textured_primitives_have_texcoord_0"],
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
        combined_or_scene, meshes = self._flatten_loaded_asset(loaded)

        total_verts = sum(len(m.vertices) for m in meshes)
        total_faces = sum(len(m.faces) for m in meshes)

        if total_verts == 0 or total_faces == 0:
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
        
        # Accessor check (POSITION, NORMAL, TEXCOORD_0)
        # STRICT: Read from raw GLB JSON, not trimesh properties.
        strict_accessors = inspect_glb_primitive_attributes(glb_path)
        has_position_accessor = strict_accessors["has_position_accessor"]
        has_normal_accessor = strict_accessors["has_normal_accessor"]
        has_texcoord_0_accessor = strict_accessors["has_texcoord_0_accessor"]
                    
        # NOTE: mesh.split() runs connected-component analysis which is O(faces).
        # For very high polycount meshes (>500k faces) this can be slow.
        # We skip the expensive split for such meshes and report component_count=1.
        _SPLIT_FACE_LIMIT = 500_000
        if total_faces <= _SPLIT_FACE_LIMIT:
            component_count = sum(len(m.split(only_watertight=False)) for m in meshes)
        else:
            component_count = len(meshes)  # treat each geometry as one component
        bounds = combined_or_scene.bounds
        
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
            "vertex_count": total_verts,
            "face_count": total_faces,
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
            "all_primitives_have_position": bool(strict_accessors["all_primitives_have_position"]),
            "all_primitives_have_normal": bool(strict_accessors["all_primitives_have_normal"]),
            "all_textured_primitives_have_texcoord_0": bool(strict_accessors["all_textured_primitives_have_texcoord_0"]),
            # Legacy fields
            "has_position_accessor": bool(strict_accessors["all_primitives_have_position"]),
            "has_normal_accessor": bool(strict_accessors["all_primitives_have_normal"]),
            "has_texcoord_0_accessor": bool(strict_accessors["all_textured_primitives_have_texcoord_0"]),
            "primitive_attribute_report": strict_accessors["primitive_attribute_report"],
            "material_integrity_status": "present" if has_material else "missing",
            "texture_applied_successfully": bool(has_embedded_texture),
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
