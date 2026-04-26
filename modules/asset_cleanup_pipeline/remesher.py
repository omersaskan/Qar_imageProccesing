import fast_simplification
import numpy as np
import trimesh
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger("remesher")

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
                hasattr(mesh, "visual")
                and hasattr(mesh.visual, "uv")
                and len(mesh.visual.uv) > 0
            )
            # trimesh often stores material in visual.material or visual.kind
            has_material = (
                hasattr(mesh.visual, "material") or mesh.visual.kind == "texture"
            )
        except Exception:
            pass
        return has_uv, has_material

    def process(
        self, mesh_path: str, output_path: str, profile: CleanupProfile
    ) -> Dict[str, Any]:
        """
        Reduces mesh polycount to meet profile limits.
        Attempts to preserve UVs/Materials if present.
        """
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        pre_faces = len(mesh.faces)
        target_faces = profile.face_count_limit

        has_uv_init, has_mat_init = self._inspect_visuals(mesh)
        
        decimation_status = "skipped"
        uv_preserved = has_uv_init
        material_preserved = has_mat_init

        if pre_faces > target_faces:
            decimation_status = "success"
            try:
                if has_uv_init:
                    # SPRINT 5C: Strict limit enforcement for mobile profiles
                    # trimesh.simplify_quadric_decimation preserves UVs
                    actual_target = target_faces
                    
                    logger.info(f"Decimating {pre_faces} -> {actual_target} (UV-safe)...")
                    candidate = mesh.simplify_quadric_decimation(actual_target)
                    has_uv_post, has_mat_post = self._inspect_visuals(candidate)
                    
                    if not has_uv_post or (has_mat_init and not has_mat_post):
                        logger.warning("UV-safe decimation failed visual integrity. Trying fallback...")
                        decimation_status = "failed_visual_integrity"
                        # Fallback will be handled below if still above limit
                    else:
                        mesh = candidate
                        uv_preserved = True
                        material_preserved = True
                else:
                    # Untextured assets can use fast_simplification directly
                    points = mesh.vertices.astype(np.float32)
                    faces = mesh.faces.astype(np.uint32)
                    reduction_ratio = 1.0 - (target_faces / max(pre_faces, 1))
                    new_vertices, new_faces = fast_simplification.simplify(points, faces, max(0.0, reduction_ratio))
                    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
            except Exception as e:
                logger.error(f"Decimation error: {e}")
                decimation_status = f"error: {str(e)}"

        # FINAL PASS: Hard enforcement if still above limit or if UV-safe decimation was skipped/failed
        if len(mesh.faces) > target_faces:
            logger.warning(f"Mesh still above limit ({len(mesh.faces)} > {target_faces}). Forced fast_simplification fallback.")
            try:
                points = mesh.vertices.astype(np.float32)
                faces = mesh.faces.astype(np.uint32)
                # SPRINT 5C: fast_simplification uses REDUCTION ratio in this environment
                reduction_ratio = 1.0 - (target_faces / max(len(mesh.faces), 1))
                new_vertices, new_faces = fast_simplification.simplify(points, faces, max(0.0, reduction_ratio))
                mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
                decimation_status = "success_fallback"
                # UVs are likely lost or corrupted here if they existed
                uv_preserved, material_preserved = self._inspect_visuals(mesh)
            except Exception as e:
                logger.error(f"Fallback decimation failed: {e}")

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
