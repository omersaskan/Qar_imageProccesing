import fast_simplification
import numpy as np
import trimesh
from typing import Dict, Any, Tuple

from .profiles import CleanupProfile


class Remesher:
    def __init__(self):
        pass

    def _inspect_visuals(self, mesh: trimesh.Trimesh) -> Tuple[bool, bool]:
        """
        Returns (has_uv, has_material)
        """
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
            
        return bool(has_uv), bool(has_material)

    def process(self, input_path: str, output_path: str, profile: CleanupProfile) -> Dict[str, Any]:
        """
        Simplify geometry while trying not to destroy UV/material data.
        Returns detailed decimation stats.
        """
        mesh = trimesh.load(input_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        pre_faces = len(mesh.faces)
        if pre_faces == 0:
            mesh.export(output_path)
            return {
                "pre_decimation_face_count": 0,
                "post_decimation_face_count": 0,
                "decimation_ratio": 1.0,
                "uv_preserved": True,
                "material_preserved": True,
                "texture_preserved": True,
                "decimation_status": "skipped_empty"
            }

        mesh.fill_holes()
        mesh.process(validate=True)

        target_faces = int(profile.target_polycount)
        has_uv_init, has_mat_init = self._inspect_visuals(mesh)
        
        decimation_status = "not_needed"
        uv_preserved = has_uv_init
        material_preserved = has_mat_init
        texture_preserved = True # Assumed if material/uv ok

        if pre_faces > target_faces:
            decimation_status = "success"
            try:
                if has_uv_init:
                    # For textured meshes, we use trimesh simplify which attempts to preserve UVs
                    # We use a conservative target if it's very high
                    actual_target = max(target_faces, int(pre_faces * 0.5)) if profile.name == "mobile_high" else target_faces
                    
                    candidate = mesh.simplify_quadric_decimation(actual_target)
                    has_uv_post, has_mat_post = self._inspect_visuals(candidate)
                    
                    if not has_uv_post or (has_mat_init and not has_mat_post):
                        decimation_status = "failed_visual_integrity"
                        uv_preserved = has_uv_post
                        material_preserved = has_mat_post
                        # We do NOT apply decimation if it breaks visuals for textured assets
                    else:
                        mesh = candidate
                        uv_preserved = True
                        material_preserved = True
                else:
                    # Untextured assets can use fast_simplification
                    points = mesh.vertices.astype(np.float32)
                    faces = mesh.faces.astype(np.uint32)
                    ratio = target_faces / max(pre_faces, 1)
                    new_vertices, new_faces = fast_simplification.simplify(points, faces, ratio)
                    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
            except Exception as e:
                decimation_status = f"error: {str(e)}"

        mesh.process(validate=True)
        mesh.remove_unreferenced_vertices()
        mesh.export(output_path)
        
        post_faces = len(mesh.faces)
        
        return {
            "pre_decimation_face_count": int(pre_faces),
            "post_decimation_face_count": int(post_faces),
            "decimation_ratio": float(post_faces / max(pre_faces, 1)),
            "uv_preserved": uv_preserved,
            "material_preserved": material_preserved,
            "texture_preserved": uv_preserved and material_preserved,
            "decimation_status": decimation_status
        }

    def pre_decimate(self, input_path: str, output_path: str, target_faces: int) -> Dict[str, Any]:
        """
        Fast geometry-only simplification for huge raw meshes.
        Uses fast_simplification to avoid the overhead of trimesh isolation/processing.
        """
        try:
            # We still use trimesh to load but with process=False to be as fast as possible
            mesh = trimesh.load(input_path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            pre_faces = len(mesh.faces)
            if pre_faces <= target_faces:
                mesh.export(output_path)
                return {
                    "pre_decimation_face_count": pre_faces,
                    "post_decimation_face_count": pre_faces,
                    "status": "skipped_already_small"
                }

            points = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.uint32)
            ratio = target_faces / max(pre_faces, 1)
            
            new_vertices, new_faces = fast_simplification.simplify(points, faces, ratio)
            new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
            new_mesh.export(output_path)
            
            return {
                "pre_decimation_face_count": int(pre_faces),
                "post_decimation_face_count": int(len(new_faces)),
                "status": "success"
            }
        except MemoryError:
            return {
                "pre_decimation_face_count": 0,
                "post_decimation_face_count": 0,
                "status": "failed_memory_limit",
                "error": "System ran out of memory while loading huge raw mesh."
            }
        except Exception as e:
            return {
                "pre_decimation_face_count": 0,
                "post_decimation_face_count": 0,
                "status": "failed_error",
                "error": str(e)
            }
