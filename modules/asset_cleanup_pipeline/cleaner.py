from __future__ import annotations
import os
import traceback
from pathlib import Path
from typing import Tuple, Dict, Optional, Any, List

import trimesh
import numpy as np

from modules.operations.logging_config import get_component_logger
from .profiles import PROFILES, CleanupProfileType
from .remesher import Remesher
from .alignment import AlignmentProcessor
from .bbox import BBoxExtractor
from .normalizer import Normalizer, NormalizedMetadata
from .isolation import MeshIsolator
from modules.operations.settings import settings
from modules.utils.mesh_inspection import get_mesh_stats_cheaply

logger = get_component_logger("cleaner")


class AssetCleaner:
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.cleaned_root = self.data_root / "cleaned"
        self.cleaned_root.mkdir(parents=True, exist_ok=True)

        self.isolator = MeshIsolator()
        self.remesher = Remesher()
        self.alignment = AlignmentProcessor()
        self.bbox_extractor = BBoxExtractor()
        self.normalizer = Normalizer()

    def _inspect_visuals(self, mesh_path: str) -> Tuple[bool, bool]:
        try:
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            has_uv = False
            has_material = False
            try:
                has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
            except Exception: pass
            try:
                has_material = hasattr(mesh.visual, "material") and mesh.visual.material is not None
            except Exception: pass
            return bool(has_uv), bool(has_material)
        except Exception:
            return False, False

    def _is_valid_textured_obj_bundle(
        self, raw_mesh_path: str, raw_texture_path: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        mesh_path = Path(raw_mesh_path)
        if mesh_path.suffix.lower() != ".obj":
            return False, f"Not an OBJ file: {mesh_path.suffix}", None
        if not mesh_path.exists():
            return False, f"Mesh file missing: {raw_mesh_path}", None
        vt_found = False
        mtllib_filename = None
        try:
            with open(mesh_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("vt "): vt_found = True
                    if stripped.startswith("mtllib "):
                        parts = stripped.split(None, 1)
                        if len(parts) > 1: mtllib_filename = parts[1].strip()
                    if vt_found and mtllib_filename: break
        except Exception as e:
            return False, f"Error reading OBJ: {e}", None
        if not vt_found: return False, "OBJ missing 'vt' lines (no UVs)", None
        if not mtllib_filename: return False, "OBJ missing 'mtllib' reference", None
        mtl_path = mesh_path.parent / mtllib_filename
        if not mtl_path.exists(): return False, f"MTL file missing: {mtllib_filename}", None
        map_kd_filename = None
        try:
            with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip().startswith("map_Kd "):
                        parts = line.strip().split(None, 1)
                        if len(parts) > 1: map_kd_filename = parts[1].strip()
                        break
        except Exception as e: return False, f"Error reading MTL: {e}", None
        if not map_kd_filename: return False, "MTL missing 'map_Kd' reference", None
        resolved_tex_path = None
        if raw_texture_path and Path(raw_texture_path).exists():
            resolved_tex_path = raw_texture_path
        else:
            potential_tex = mtl_path.parent / map_kd_filename
            if potential_tex.exists(): resolved_tex_path = str(potential_tex)
        if not resolved_tex_path: return False, f"Texture file missing: {map_kd_filename}", None
        return True, "Valid textured OBJ bundle", resolved_tex_path

    def _run_texture_safe_copy(
        self,
        raw_mesh_path: str,
        raw_texture_path: str,
        output_dir: Path,
        cameras: Optional[List[Dict]] = None,
        masks: Optional[Dict[str, np.ndarray]] = None,
        point_cloud: Optional[trimesh.points.PointCloud] = None,
    ) -> dict:
        raw_mesh = Path(raw_mesh_path)
        raw_tex = Path(raw_texture_path)
        if not raw_mesh_path.lower().endswith(".obj"):
             raise ValueError(f"texture_safe_copy requires .obj input, got {raw_mesh.suffix}")
        if not raw_tex.exists():
            raise ValueError(f"Texture missing: {raw_texture_path}")
        
        # Aligned OBJ and normalization
        cleaned_mesh_path = output_dir / "cleaned_mesh.obj"
        
        try:
            mesh = trimesh.load(raw_mesh, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            isolated_mesh, isolation_stats = self.isolator.isolate_product(
                mesh,
                cameras=cameras,
                masks=masks,
                point_cloud=point_cloud,
                output_dir=output_dir
            )
            if isolation_stats.get("object_isolation_status") != "success":
                raise ValueError(f"Object isolation failed: {isolation_stats.get('object_isolation_status')}")
            isolated_debug_obj = output_dir / "debug_isolated_mesh.obj"
            isolated_mesh.export(str(isolated_debug_obj))
            work_mesh = isolated_debug_obj
        except Exception as e:
            logger.error("Isolation failed during texture_safe_copy: %s\n%s", e, traceback.format_exc())
            return {
                "cleanup_mode": "failed",
                "object_isolation_status": "failed",
                "unsafe_scene_copy_forbidden": True,
                "error": str(e)
            }

        pivot_offset, bbox_min, bbox_max = self._safe_align_obj(work_mesh, cleaned_mesh_path)
        
        import shutil
        cleaned_tex_path = output_dir / raw_tex.name
        shutil.copy2(raw_tex, cleaned_tex_path)
        
        mtl_filename = "material.mtl"
        cleaned_mtl_path = output_dir / mtl_filename
        with open(cleaned_mtl_path, "w", encoding="utf-8") as f:
            f.write(f"newmtl material_0\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\nd 1.0\nillum 2\nmap_Kd {raw_tex.name}\n")
            
        face_count = 0
        with open(cleaned_mesh_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("f "): face_count += 1
        
        has_uv, has_mat = self._inspect_visuals(str(cleaned_mesh_path))
        texture_applied = bool(has_uv and cleaned_tex_path.exists())
        
        # Strict Delivery Gate
        delivery_ready = True
        if settings.require_textured_output or settings.fail_on_texture_missing:
            if not texture_applied:
                delivery_ready = False
        if settings.fail_on_uv_missing and not has_uv:
            delivery_ready = False
        
        # SPRINT 5 Fix: If isolation is only geometric, it's not production-grade (Review Ready only)
        if isolation_stats.get("object_isolation_method") == "geometric_only":
            delivery_ready = False # In our internal logic, False means 'review_ready' or 'failed' depending on other flags

        # Phase 6.1: SAM2 outputs must be review_ready
        if isolation_stats.get("used_sam2"):
            delivery_ready = False

        # ROOT CAUSE FIX: Semantic Quality Gate
        iso_quality_status, iso_quality_msg = self._evaluate_isolation_quality(isolation_stats)
        if iso_quality_status == "fail":
            delivery_ready = False
            isolation_stats["quality_status"] = "fail"
            isolation_stats["quality_reason"] = iso_quality_msg
            logger.warning("[%s] Isolation quality gate FAILED: %s", output_dir.name, iso_quality_msg)
        elif iso_quality_status == "review":
            delivery_ready = False
            isolation_stats["quality_status"] = "review"
            isolation_stats["quality_reason"] = iso_quality_msg
            logger.info("[%s] Isolation quality gate flagged for REVIEW: %s", output_dir.name, iso_quality_msg)
        else:
            isolation_stats["quality_status"] = "pass"

        return {
            "cleanup_mode": "texture_safe_copy",
            "uv_preserved": True,
            "material_preserved": True,
            "cleaned_mesh_path": str(cleaned_mesh_path),
            "cleaned_texture_path": str(cleaned_tex_path),
            "final_polycount": int(face_count),
            "pivot_offset": pivot_offset,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "isolation": isolation_stats,
            "decimation": {"decimation_status": "skipped_safe_copy", "uv_preserved": True, "material_preserved": True},
            "texture_input_mesh_path": str(work_mesh),
            "unsafe_scene_copy_forbidden": False,
            "has_uv": has_uv,
            "has_material": has_mat,
            "texture_applied": texture_applied,
            "delivery_ready": delivery_ready,
        }

    def _safe_align_obj(self, input_path: Path, output_path: Path) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        vertices = []
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4: vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not vertices: raise ValueError("No vertices found in OBJ")
        import numpy as np
        v_np = np.array(vertices)
        v_min, v_max, v_centroid = v_np.min(axis=0), v_np.max(axis=0), v_np.mean(axis=0)
        shift_x, shift_y, shift_z = -float(v_centroid[0]), -float(v_centroid[1]), -float(v_min[2])
        pivot_offset = {"x": shift_x, "y": shift_y, "z": shift_z}
        usemtl_written = False
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        nx, ny, nz = float(parts[1]) + shift_x, float(parts[2]) + shift_y, float(parts[3]) + shift_z
                        f_out.write(f"v {nx:.6f} {ny:.6f} {nz:.6f}\n")
                    else: f_out.write(line)
                elif line.startswith("usemtl "):
                    f_out.write("usemtl material_0\n")
                    usemtl_written = True
                elif line.startswith("f "):
                    if not usemtl_written: f_out.write("usemtl material_0\n"); usemtl_written = True
                    f_out.write(line)
                else: f_out.write(line)
            if not usemtl_written: f_out.write("usemtl material_0\n")
        bbox_min = {"x": float(v_min[0] + shift_x), "y": float(v_min[1] + shift_y), "z": 0.0}
        bbox_max = {"x": float(v_max[0] + shift_x), "y": float(v_max[1] + shift_y), "z": float(v_max[2] + shift_z)}
        return pivot_offset, bbox_min, bbox_max

    def process_cleanup(
        self,
        job_id: str,
        raw_mesh_path: str,
        profile_type: CleanupProfileType = CleanupProfileType.MOBILE_HIGH,
        raw_texture_path: Optional[str] = None,
        cameras: Optional[List[Dict]] = None,
        masks: Optional[Dict[str, np.ndarray]] = None,
        point_cloud: Optional[trimesh.points.PointCloud] = None,
    ) -> Tuple[NormalizedMetadata, dict, str]:
        if not os.path.exists(raw_mesh_path):
            raise FileNotFoundError(f"Raw mesh not found: {raw_mesh_path}")

        profile = PROFILES[profile_type]
        job_cleaned_dir = self.cleaned_root / job_id
        job_cleaned_dir.mkdir(parents=True, exist_ok=True)

        # SPRINT 5: Auto-switch to TEXTURE_SAFE_COPY if it's a valid textured bundle and not already forced
        if profile_type != CleanupProfileType.TEXTURE_SAFE_COPY and raw_texture_path:
             valid, _, _ = self._is_valid_textured_obj_bundle(raw_mesh_path, raw_texture_path)
             if valid:
                 logger.info("[%s] Valid textured OBJ bundle detected. Auto-switching to TEXTURE_SAFE_COPY profile.", job_id)
                 profile_type = CleanupProfileType.TEXTURE_SAFE_COPY

        if profile_type == CleanupProfileType.TEXTURE_SAFE_COPY:
            if not raw_texture_path: raise ValueError("TEXTURE_SAFE_COPY requires texture")
            valid, msg, resolved_tex = self._is_valid_textured_obj_bundle(raw_mesh_path, raw_texture_path)
            if not valid:
                raise ValueError(msg)
            stats = self._run_texture_safe_copy(
                raw_mesh_path, 
                resolved_tex, 
                job_cleaned_dir,
                cameras=cameras,
                masks=masks,
                point_cloud=point_cloud
            )
            if stats.get("cleanup_mode") == "failed": raise RuntimeError(f"texture_safe_copy failed: {stats.get('error')}")
            metadata = self.normalizer.generate_metadata(stats["bbox_min"], stats["bbox_max"], stats["pivot_offset"], stats["final_polycount"])
            metadata_path = job_cleaned_dir / "normalized_metadata.json"
            self.normalizer.save_metadata(metadata, str(metadata_path))
            stats["metadata_path"] = str(metadata_path)
            stats["delivery_profile"] = profile.name
            return metadata, stats, stats["cleaned_mesh_path"]

        isolation_debug_path = job_cleaned_dir / "debug_isolated_mesh.obj"
        cleaned_mesh_path = job_cleaned_dir / "cleaned_mesh.obj"
        metadata_path = job_cleaned_dir / "normalized_metadata.json"
        pre_aligned_path = job_cleaned_dir / "pre_aligned_mesh.obj"

        logger.info("[%s] Entering cleanup stage for %s", job_id, raw_mesh_path)
        
        # 1) Pre-cleanup mesh budget gate (Cheap inspection)
        raw_stats = get_mesh_stats_cheaply(raw_mesh_path)
        raw_faces = raw_stats["face_count"]
        raw_verts = raw_stats["vertex_count"]
        
        logger.info("[%s] Raw mesh inspection: faces=%d, vertices=%d", job_id, raw_faces, raw_verts)
        
        oversized_raw_mesh = False
        work_mesh_path = raw_mesh_path
        
        if raw_faces > settings.recon_mesh_hard_limit_faces:
            logger.error("[%s] Mesh is far beyond hard limit (%d faces). Failing fast.", job_id, raw_faces)
            # Return specialized status for the runner/worker to handle
            return None, {
                "status": "failed_oversized_mesh", 
                "raw_faces": raw_faces,
                "reason": f"Mesh exceeds hard limit of {settings.recon_mesh_hard_limit_faces} faces. Lower RECON_POISSON_DEPTH or capture a smaller area."
            }, ""

        if raw_faces > settings.recon_mesh_budget_faces:
            logger.warning("[%s] Mesh is oversized (%d faces). Triggering pre-decimation gate.", job_id, raw_faces)
            oversized_raw_mesh = True
            
            # SPRINT 5: Safe threshold check for Python load
            if raw_faces > settings.max_faces_python_decimation:
                logger.error("[%s] Mesh faces (%d) exceed safe Python load limit (%d). Skipping pre-decimation to avoid OOM.", 
                             job_id, raw_faces, settings.max_faces_python_decimation)
                return None, {
                    "status": "failed_memory_limit",
                    "cleanup_failure_type": "pre_decimation_avoided_oversized",
                    "raw_mesh_faces": raw_faces,
                    "raw_mesh_vertices": raw_verts,
                    "recommended_poisson_depth": settings.recon_poisson_depth - 1,
                    "recommended_poisson_trim": settings.recon_poisson_trim + 1,
                    "retryable_from_fused_ply": True,
                    "reason": f"Mesh faces ({raw_faces}) exceed safe Python load limit. Lower Poisson depth to reduce raw mesh density."
                }, ""

            pre_decimate_path = job_cleaned_dir / "pre_decimated_raw.ply"
            logger.info("[%s] Starting pre-decimation (target=%d faces)...", job_id, settings.recon_pre_cleanup_target_faces)
            
            pre_dec_stats = self.remesher.pre_decimate(
                raw_mesh_path, 
                str(pre_decimate_path), 
                settings.recon_pre_cleanup_target_faces
            )
            
            if pre_dec_stats["status"] != "success" and pre_dec_stats["status"] != "skipped_already_small":
                logger.error("[%s] Pre-decimation failed: %s", job_id, pre_dec_stats.get("error"))
                failure_type = "pre_decimation_oom" if pre_dec_stats["status"] == "failed_memory_limit" else "pre_decimation_error"
                return None, {
                    "status": "failed_memory_limit" if pre_dec_stats["status"] == "failed_memory_limit" else "failed_oversized_mesh",
                    "cleanup_failure_type": failure_type,
                    "raw_mesh_faces": raw_faces,
                    "raw_mesh_vertices": raw_verts,
                    "recommended_poisson_depth": settings.recon_poisson_depth - 1,
                    "recommended_poisson_trim": settings.recon_poisson_trim + 1,
                    "retryable_from_fused_ply": True,
                    "reason": f"Pre-decimation failed: {pre_dec_stats.get('error')}. Lower Poisson depth to reduce raw mesh density."
                }, ""

            logger.info("[%s] Pre-decimation completed: %d -> %d faces", 
                        job_id, pre_dec_stats["pre_decimation_face_count"], pre_dec_stats["post_decimation_face_count"])
            work_mesh_path = str(pre_decimate_path)

        try:
            # 2) Object Isolation
            logger.info("[%s] Starting isolation on %s. Data Support: cameras=%s, masks=%s, point_cloud=%s", 
                        job_id, work_mesh_path, bool(cameras), bool(masks), bool(point_cloud))
            mesh = trimesh.load(work_mesh_path, process=False)
            if isinstance(mesh, trimesh.Scene): mesh = mesh.dump(concatenate=True)
            if len(mesh.vertices) == 0: raise ValueError("Empty mesh")

            isolated_mesh, isolation_stats = self.isolator.isolate_product(
                mesh, 
                cameras=cameras, 
                masks=masks, 
                point_cloud=point_cloud,
                output_dir=job_cleaned_dir
            )
            if len(isolated_mesh.faces) == 0: raise ValueError(f"Isolation failed: {isolation_stats.get('object_isolation_status')}")
            isolated_mesh.export(str(isolation_debug_path))
            logger.info("[%s] Isolation completed. Method: %s, faces: %d", 
                        job_id, isolation_stats.get("object_isolation_method"), len(isolated_mesh.faces))

            # 3) Remeshing/Decimation
            logger.info("[%s] Starting remesher (profile=%s)", job_id, profile.name)
            decimation_stats = self.remesher.process(str(isolation_debug_path), str(pre_aligned_path), profile)
            final_polycount = decimation_stats["post_decimation_face_count"]
            logger.info("[%s] Remesher completed. Final faces: %d", job_id, final_polycount)

            # 4) Alignment & BBox
            logger.info("[%s] Starting alignment and normalization", job_id)
            _, pivot_offset = self.alignment.align_to_ground(str(pre_aligned_path), str(cleaned_mesh_path))
            bbox_min, bbox_max = self.bbox_extractor.extract(str(cleaned_mesh_path))

            metadata = self.normalizer.generate_metadata(bbox_min, bbox_max, pivot_offset, final_polycount)
            self.normalizer.save_metadata(metadata, str(metadata_path))

            has_uv, has_material = self._inspect_visuals(str(cleaned_mesh_path))
            texture_applied = bool(raw_texture_path and has_uv)
            
            # Strict Delivery Gate
            delivery_ready = True
            if settings.require_textured_output or settings.fail_on_texture_missing:
                if not raw_texture_path:
                    delivery_ready = False
            if settings.fail_on_uv_missing and not has_uv:
                delivery_ready = False
            
            if isolation_stats.get("object_isolation_method") == "geometric_only":
                delivery_ready = False
            
            # Phase 6.1: SAM2 outputs must be review_ready
            if isolation_stats.get("used_sam2"):
                delivery_ready = False
            
            # ROOT CAUSE FIX: Semantic Quality Gate
            iso_quality_status, iso_quality_msg = self._evaluate_isolation_quality(isolation_stats)
            if iso_quality_status == "fail":
                delivery_ready = False
                isolation_stats["quality_status"] = "fail"
                isolation_stats["quality_reason"] = iso_quality_msg
                logger.warning("[%s] Isolation quality gate FAILED: %s", job_id, iso_quality_msg)
            elif iso_quality_status == "review":
                delivery_ready = False
                isolation_stats["quality_status"] = "review"
                isolation_stats["quality_reason"] = iso_quality_msg
                logger.info("[%s] Isolation quality gate flagged for REVIEW: %s", job_id, iso_quality_msg)
            else:
                isolation_stats["quality_status"] = "pass"
            
            logger.info("[%s] Cleanup pipeline finished. Output: %s, delivery_ready=%s", job_id, cleaned_mesh_path, delivery_ready)
            
            cleanup_stats = {
                "isolation": isolation_stats,
                "decimation": decimation_stats,
                "final_polycount": int(final_polycount),
                "poly_count": int(final_polycount), # Alias for validator
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "bbox": {"x": bbox_max["x"]-bbox_min["x"], "y": bbox_max["y"]-bbox_min["y"], "z": bbox_max["z"]-bbox_min["z"]},
                "ground_offset": abs(pivot_offset["z"]),
                "cleaned_mesh_path": str(cleaned_mesh_path),
                "metadata_path": str(metadata_path),
                "cleaned_texture_path": str(raw_texture_path) if raw_texture_path else None,
                "texture_path": str(raw_texture_path) if raw_texture_path else None, # Alias
                "delivery_profile": profile.name,
                "has_uv": bool(has_uv),
                "has_material": bool(has_material),
                "texture_applied": texture_applied,
                "delivery_ready": delivery_ready,
                "texture_input_mesh_path": str(isolation_debug_path),
                "pre_aligned_mesh_path": str(pre_aligned_path),
                "oversized_raw_mesh": oversized_raw_mesh,
                "raw_mesh_faces": raw_faces,
                "raw_mesh_vertices": raw_verts,
            }
            return metadata, cleanup_stats, str(cleaned_mesh_path)

        except MemoryError:
            logger.error("[%s] Cleanup failed with MemoryError (OOM) during part isolation or remeshing.", job_id)
            return None, {
                "status": "failed_memory_limit",
                "cleanup_failure_type": "post_decimation_oom",
                "raw_mesh_faces": raw_faces,
                "raw_mesh_vertices": raw_verts,
                "recommended_poisson_depth": settings.recon_poisson_depth_retry if raw_faces > 1000000 else 8,
                "retryable_from_fused_ply": True,
                "reason": "System ran out of memory during mesh cleanup. Lowering Poisson density is recommended."
            }, ""

    def _evaluate_isolation_quality(self, stats: Dict[str, Any]) -> Tuple[str, str]:
        """
        Hard gate for semantic isolation quality.
        Returns (status, message) where status is 'pass', 'review', or 'fail'.
        """
        primary_share = stats.get("primary_face_share", 1.0)
        kept_count = stats.get("kept_component_count", 1)
        largest_share = stats.get("largest_kept_component_share", 1.0)
        
        # Rule 1: Primary component must be dominant in the isolated result
        if primary_share < 0.50:
            return "fail", f"Primary component is too small ({primary_share:.2%} of isolated mesh). Fragmented result."
            
        # Rule 2: If multiple components are kept, one should be clearly dominant
        if kept_count > 1 and largest_share < 0.70:
            return "review", f"Multiple components kept ({kept_count}) without a dominant one (largest={largest_share:.2%}). Contamination likely."
            
        # Rule 3: Isolation confidence (total_score of primary)
        confidence = stats.get("isolation_confidence", 0.0)
        if confidence < 0.35:
            return "review", f"Low isolation confidence ({confidence:.2f})."
            
        return "pass", ""