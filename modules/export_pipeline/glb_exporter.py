import os
import struct
import json
from modules.operations.settings import settings
import logging
from typing import Dict, Any, Optional, List, Tuple
import trimesh
import numpy as np
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata

logger = logging.getLogger("glb_exporter")

def inspect_glb_primitive_attributes(glb_path: str) -> Dict[str, Any]:
    """
    STRICT GLB inspector that reads the raw JSON chunk to verify accessors.
    """
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB file not found for inspection: {glb_path}")

    with open(glb_path, "rb") as f:
        magic = f.read(4)
        if magic != b"glTF":
            f.seek(0)
            try:
                gltf = json.load(f)
            except Exception:
                raise ValueError(f"Not a valid glTF/GLB file: {glb_path}")
        else:
            f.read(8) # version + total length
            found_json = False
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

            is_textured = False
            mat_idx = prim.get("material")
            if mat_idx is not None and isinstance(mat_idx, int) and mat_idx < len(materials):
                mat = materials[mat_idx]
                pbr = mat.get("pbrMetallicRoughness", {})
                if any(k in pbr for k in ["baseColorTexture", "metallicRoughnessTexture"]) or \
                   any(k in mat for k in ["normalTexture", "occlusionTexture", "emissiveTexture"]):
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

            primitive_reports.append({
                "mesh_index": mesh_idx,
                "primitive_index": prim_idx,
                "attributes": list(attrs.keys()),
                "has_position": has_pos,
                "has_normal": has_norm,
                "has_texcoord_0": has_uv,
                "is_textured": is_textured,
                "missing_attributes": missing,
            })

    return {
        "all_primitives_have_position": all_primitives_have_position,
        "all_primitives_have_normal": all_primitives_have_normal,
        "all_textured_primitives_have_texcoord_0": all_textured_primitives_have_texcoord_0,
        "primitive_attribute_report": primitive_reports,
        "texture_count": len(gltf.get("textures", [])),
        "material_count": len(gltf.get("materials", [])),
    }

class GLBExporter:
    def __init__(self):
        pass

    def _inspect_visuals(self, mesh: trimesh.Trimesh) -> Dict[str, bool]:
        has_uv = False
        has_material = False
        try:
            has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
        except Exception: pass
        try:
            has_material = hasattr(mesh.visual, "material") and mesh.visual.material is not None
        except Exception: pass
        return {"has_uv": bool(has_uv), "has_material": bool(has_material)}

    def export(
        self,
        mesh_path: str,
        output_path: str,
        profile_name: str = "raw_archive",
        texture_path: Optional[str] = None,
        smoothing_mode: str = "none",
        metadata: Optional[NormalizedMetadata] = None,
    ) -> Dict[str, Any]:
        """
        Final GLB Export with Delivery Gates.
        """
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Source mesh not found: {mesh_path}")

        loaded = trimesh.load(mesh_path)
        if isinstance(loaded, trimesh.Scene):
            meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        else:
            meshes = [loaded]

        # 1. Normals & Smoothing
        for m in meshes:
            if smoothing_mode == "conservative":
                m.fix_normals(multibody=True)
                m.process(validate=True)
            elif smoothing_mode == "photogrammetry_smooth":
                # Basic Laplacian-style smoothing without destroying UVs (trimesh filter)
                trimesh.smoothing.filter_laplacian(m, lamb=0.5, iterations=5)
            
            # Mandatory: Ensure vertex normals are materialized
            m.fix_normals(multibody=True)
            _ = m.vertex_normals 

        # 2. Texture Injection
        texture_applied = False
        if texture_path and os.path.exists(texture_path):
            try:
                from PIL import Image
                with Image.open(texture_path) as img:
                    tex_image = img.convert("RGBA").copy()
                
                material = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=tex_image,
                    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                    doubleSided=True,
                )
                
                for m in meshes:
                    if hasattr(m.visual, "uv") and m.visual.uv is not None and len(m.visual.uv) > 0:
                        m.visual = trimesh.visual.TextureVisuals(uv=m.visual.uv, material=material)
                        texture_applied = True
            except Exception as e:
                logger.warning("Texture application failed: %s", e)

        # 3. Export
        if len(meshes) > 1:
            scene = trimesh.Scene(meshes)
            glb_bytes = scene.export(file_type="glb")
        else:
            glb_bytes = meshes[0].export(file_type="glb")

        # SPRINT 5: Fix 7 — GLB Compliance (bufferView targets)
        glb_bytes = self._fix_glb_compliance(glb_bytes)

        with open(output_path, "wb") as f:
            f.write(glb_bytes)

        # 4. Optimization Hooks (Placeholders)
        opt_report = self._apply_optional_optimizations(output_path)

        # 5. Final Strict Inspection
        inspection = self.inspect_exported_asset(output_path)
        
        # Delivery Gate logic
        all_accessors = (
            inspection["all_primitives_have_position"] and
            inspection["all_primitives_have_normal"] and
            inspection["all_textured_primitives_have_texcoord_0"]
        )
        
        # Texture Gate: If texture was requested but not applied, it's a failure
        texture_success = True
        if texture_path and not texture_applied:
            texture_success = False
        
        if inspection["texture_count"] == 0 and (texture_path or settings.require_textured_output):
            texture_success = False
            
        # Texture Quality Gate
        texture_quality_pass = True
        if inspection.get("highest_black_pixel_ratio", 0.0) > settings.max_empty_texture_ratio:
            texture_quality_pass = False
            
        structural_export_ready = all_accessors and texture_success and texture_quality_pass
        
        if not all_accessors:
            export_status = "failed_validation"
        elif not texture_success:
            export_status = "failed_texture_application"
        elif not texture_quality_pass:
            export_status = "failed_texture_quality"
        else:
            export_status = "success"

        result = {
            "format": "GLB",
            "filesize": os.path.getsize(output_path),
            "profile": profile_name,
            "smoothing_mode": smoothing_mode,
            "texture_applied": texture_applied,
            "optimization_hooks": opt_report,
            "structural_export_ready": structural_export_ready,
            "export_status": export_status,
            **inspection
        }

        if metadata:
            result.update({
                "bbox_min": metadata.bbox_min,
                "bbox_max": metadata.bbox_max,
                "final_polycount": metadata.final_polycount
            })

        return result

    def _fix_glb_compliance(self, glb_bytes: bytes) -> bytes:
        """
        Manually injects bufferView.target into the GLB JSON chunk.
        - POSITION/NORMAL/TEXCOORD_0 => 34962 (ARRAY_BUFFER)
        - indices => 34963 (ELEMENT_ARRAY_BUFFER)
        """
        if len(glb_bytes) < 20: return glb_bytes
        
        # Read header
        magic = glb_bytes[:4]
        if magic != b"glTF": return glb_bytes
        
        # Read JSON chunk
        json_len = struct.unpack("<I", glb_bytes[12:16])[0]
        json_type = glb_bytes[16:20]
        if json_type != b"JSON": return glb_bytes
        
        json_data = json.loads(glb_bytes[20:20+json_len].decode("utf-8"))
        
        # Map accessors to bufferViews
        bv_targets = {}
        
        # Attributes targets
        meshes = json_data.get("meshes", [])
        for m in meshes:
            for p in m.get("primitives", []):
                for attr, acc_idx in p.get("attributes", {}).items():
                    if attr in ["POSITION", "NORMAL", "TEXCOORD_0"]:
                        acc = json_data.get("accessors", [])[acc_idx]
                        bv_idx = acc.get("bufferView")
                        if bv_idx is not None:
                            bv_targets[bv_idx] = 34962 # ARRAY_BUFFER
                
                # Indices target
                indices_idx = p.get("indices")
                if indices_idx is not None:
                    acc = json_data.get("accessors", [])[indices_idx]
                    bv_idx = acc.get("bufferView")
                    if bv_idx is not None:
                        bv_targets[bv_idx] = 34963 # ELEMENT_ARRAY_BUFFER

        # Apply targets to bufferViews
        bvs = json_data.get("bufferViews", [])
        for i, target in bv_targets.items():
            if i < len(bvs):
                bvs[i]["target"] = target
        
        # Re-encode JSON
        new_json_bytes = json.dumps(json_data, separators=(",", ":")).encode("utf-8")
        # GLB requires JSON chunk to be 4-byte aligned
        while len(new_json_bytes) % 4 != 0:
            new_json_bytes += b" "
        
        new_json_len = len(new_json_bytes)
        
        # Assemble new GLB
        new_glb = bytearray()
        new_glb.extend(glb_bytes[:8]) # magic + version
        # placeholder for total length
        new_glb.extend(struct.pack("<I", 0)) 
        
        # JSON chunk header
        new_glb.extend(struct.pack("<I", new_json_len))
        new_glb.extend(b"JSON")
        new_glb.extend(new_json_bytes)
        
        # Append remaining chunks (binary chunk)
        pos = 20 + json_len
        new_glb.extend(glb_bytes[pos:])
        
        # Update total length
        total_len = len(new_glb)
        new_glb[8:12] = struct.pack("<I", total_len)
        
        return bytes(new_glb)

    def _apply_optional_optimizations(self, glb_path: str) -> Dict[str, str]:
        """
        Placeholder for meshopt, Draco, gltf-validator.
        """
        report = {
            "meshopt": "skipped_unavailable",
            "draco": "skipped_unavailable",
            "gltf_validator": "skipped_unavailable"
        }
        # In the future, we would call external tools here
        return report

    def inspect_exported_asset(self, glb_path: str) -> Dict[str, Any]:
        """
        Combined trimesh + strict JSON inspection + texture quality check.
        """
        strict = inspect_glb_primitive_attributes(glb_path)
        
        # Load for geometry counts and texture extraction
        loaded = trimesh.load(glb_path, force="scene")
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        
        total_verts = sum(len(m.vertices) for m in meshes)
        total_faces = sum(len(m.faces) for m in meshes)
        
        # Texture analysis
        from modules.qa_validation.texture_quality import TextureQualityAnalyzer
        from modules.operations.settings import settings
        import numpy as np
        
        analyzer = TextureQualityAnalyzer()
        highest_black_pixel_ratio = 0.0
        
        textures = []
        for m in meshes:
            if hasattr(m.visual, 'material'):
                mat = m.visual.material
                if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                    textures.append(mat.baseColorTexture)
                elif hasattr(mat, 'image') and mat.image is not None:
                    textures.append(mat.image)
        
        for tex in textures:
            try:
                img_np = np.array(tex.convert("RGB"))
                img_bgr = img_np[:, :, ::-1].copy()
                report = analyzer.analyze_image(img_bgr)
                highest_black_pixel_ratio = max(highest_black_pixel_ratio, report.get("black_pixel_ratio", 0.0))
            except Exception as e:
                logger.warning("Failed to analyze texture in GLB: %s", e)

        return {
            "final_vertex_count": total_verts,
            "final_face_count": total_faces,
            "texture_count": strict["texture_count"],
            "material_count": strict["material_count"],
            "all_primitives_have_position": strict["all_primitives_have_position"],
            "all_primitives_have_normal": strict["all_primitives_have_normal"],
            "all_textured_primitives_have_texcoord_0": strict["all_textured_primitives_have_texcoord_0"],
            "highest_black_pixel_ratio": highest_black_pixel_ratio,
            "primitive_attribute_report": strict["primitive_attribute_report"]
        }
