import os
from pathlib import Path
from typing import Tuple, Dict, Optional

import trimesh

from modules.operations.logging_config import get_component_logger
from .profiles import PROFILES, CleanupProfileType
from .remesher import Remesher
from .alignment import AlignmentProcessor
from .bbox import BBoxExtractor
from .normalizer import Normalizer, NormalizedMetadata
from .isolation import MeshIsolator

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
        """
        Returns:
            (has_uv, has_material)
        """
        try:
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

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

        except Exception:
            return False, False

    def _is_valid_textured_obj_bundle(
        self, raw_mesh_path: str, raw_texture_path: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validates if the input is a valid textured OBJ bundle.
        Returns: (is_valid, reason, resolved_texture_path)
        """
        mesh_path = Path(raw_mesh_path)
        if mesh_path.suffix.lower() != ".obj":
            return False, f"Not an OBJ file: {mesh_path.suffix}", None

        if not mesh_path.exists():
            return False, f"Mesh file missing: {raw_mesh_path}", None

        # 1. Scan OBJ for vt and mtllib
        vt_found = False
        mtllib_filename = None
        try:
            with open(mesh_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("vt "):
                        vt_found = True
                    if stripped.startswith("mtllib "):
                        parts = stripped.split(None, 1)
                        if len(parts) > 1:
                            mtllib_filename = parts[1].strip()
                    if vt_found and mtllib_filename:
                        break
        except Exception as e:
            return False, f"Error reading OBJ: {e}", None

        if not vt_found:
            return False, "OBJ missing 'vt' lines (no UVs)", None
        if not mtllib_filename:
            return False, "OBJ missing 'mtllib' reference", None

        # 2. Resolve and check MTL
        mtl_path = mesh_path.parent / mtllib_filename
        if not mtl_path.exists():
            return False, f"MTL file missing: {mtllib_filename}", None

        # 3. Scan MTL for map_Kd
        map_kd_filename = None
        try:
            with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip().startswith("map_Kd "):
                        parts = line.strip().split(None, 1)
                        if len(parts) > 1:
                            map_kd_filename = parts[1].strip()
                        break
        except Exception as e:
            return False, f"Error reading MTL: {e}", None

        if not map_kd_filename:
            return False, "MTL missing 'map_Kd' reference", None

        # 4. Resolve texture
        # If raw_texture_path is provided, we use it, otherwise we use the one from MTL
        resolved_tex_path = None
        if raw_texture_path and Path(raw_texture_path).exists():
            resolved_tex_path = raw_texture_path
        else:
            potential_tex = mtl_path.parent / map_kd_filename
            if potential_tex.exists():
                resolved_tex_path = str(potential_tex)

        if not resolved_tex_path:
            return False, f"Texture file missing: {map_kd_filename}", None

        return True, "Valid textured OBJ bundle", resolved_tex_path

    def _run_texture_safe_copy(
        self,
        raw_mesh_path: str,
        raw_texture_path: str,
        output_dir: Path,
    ) -> dict:
        raw_mesh = Path(raw_mesh_path)
        raw_tex = Path(raw_texture_path)
        
        if not raw_mesh_path.lower().endswith(".obj"):
             raise ValueError(f"texture_safe_copy requires .obj input, got {raw_mesh.suffix}")
        
        if not raw_tex.exists():
            raise ValueError(f"Texture missing: {raw_texture_path}")

        # OBJ validation
        vt_found = False
        mtllib_found = False
        mtl_filename = None
        
        with open(raw_mesh, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("vt "):
                    vt_found = True
                if stripped.startswith("mtllib "):
                    mtllib_found = True
                    parts = stripped.split(None, 1)
                    if len(parts) > 1:
                        mtl_filename = parts[1].strip()
                if vt_found and mtllib_found:
                    break
        
        if not vt_found:
            raise ValueError("OBJ missing 'vt' lines")
        if not mtllib_found or not mtl_filename:
            raise ValueError("OBJ missing 'mtllib' line")
        
        raw_mtl = raw_mesh.parent / mtl_filename
        if not raw_mtl.exists():
            raise ValueError(f"MTL file missing: {raw_mtl}")
        
        # MTL validation
        map_kd_found = False
        with open(raw_mtl, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("map_Kd "):
                    map_kd_found = True
                    break
        
        if not map_kd_found:
            raise ValueError("MTL missing 'map_Kd' line")
            
        # Aligned OBJ and normalization
        cleaned_mesh_path = output_dir / "cleaned_mesh.obj"
        cleaned_mtl_path = output_dir / mtl_filename
        cleaned_tex_path = output_dir / raw_tex.name
        
        # 1. Perform isolation before alignment to remove table/support/islands
        import trimesh
        try:
            mesh = trimesh.load(raw_mesh, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            isolated_mesh, isolation_stats = self.isolator.isolate_product(mesh)
            
            # Export isolated mesh to a temp OBJ to preserve UVs (trimesh OBJ export handles vt)
            isolated_temp_obj = output_dir / "isolated_temp.obj"
            isolated_mesh.export(str(isolated_temp_obj))
            
            # Use the isolated mesh for subsequent alignment
            work_mesh = isolated_temp_obj
        except Exception as e:
            logger.warning("Isolation failed during texture_safe_copy, falling back to raw: %s", e)
            work_mesh = raw_mesh
            isolation_stats = {"isolation_error": str(e)}

        # 2. Perform safe alignment (translates vertices, computes bbox and pivot offset)
        pivot_offset, bbox_min, bbox_max = self._safe_align_obj(work_mesh, cleaned_mesh_path)
        
        # Cleanup temp
        if work_mesh != raw_mesh and work_mesh.exists():
            work_mesh.unlink()

        # 3. OBJ usemtl normalization ...
        import shutil
        shutil.copy2(raw_tex, cleaned_tex_path)
        
        # 4. Normalize MTL
        with open(cleaned_mtl_path, "w", encoding="utf-8") as f:
            f.write(f"newmtl material_0\n")
            f.write(f"Ka 1.000000 1.000000 1.000000\n")
            f.write(f"Kd 1.000000 1.000000 1.000000\n")
            f.write(f"Ks 0.000000 0.000000 0.000000\n")
            f.write(f"d 1.000000\n")
            f.write(f"illum 2\n")
            f.write(f"map_Kd {raw_tex.name}\n")
            
        # 5. Get face count
        face_count = 0
        if cleaned_mesh_path.exists():
            with open(cleaned_mesh_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("f "):
                        face_count += 1
        
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
        }

    def _safe_align_obj(self, input_path: Path, output_path: Path) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Translates vertices to ground (Z=0) and center (XY=0) without using trimesh.
        Also normalizes usemtl to material_0.
        Returns: (pivot_offset, bbox_min, bbox_max)
        """
        vertices = []
        # First pass: collect vertices and compute bounds
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

        if not vertices:
            raise ValueError("No vertices found in OBJ")

        import numpy as np
        v_np = np.array(vertices)
        v_min = v_np.min(axis=0)
        v_max = v_np.max(axis=0)
        v_centroid = v_np.mean(axis=0)

        # Shift X,Y to center, Z to ground (min_z = 0)
        shift_x = -float(v_centroid[0])
        shift_y = -float(v_centroid[1])
        shift_z = -float(v_min[2])

        pivot_offset = {"x": shift_x, "y": shift_y, "z": shift_z}
        
        # Second pass: write translated vertices and other lines
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f_in, \
             open(output_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        nx = float(parts[1]) + shift_x
                        ny = float(parts[2]) + shift_y
                        nz = float(parts[3]) + shift_z
                        f_out.write(f"v {nx:.6f} {ny:.6f} {nz:.6f}\n")
                    else:
                        f_out.write(line)
                elif line.startswith("usemtl "):
                    f_out.write("usemtl material_0\n")
                else:
                    f_out.write(line)

        bbox_min = {"x": float(v_min[0] + shift_x), "y": float(v_min[1] + shift_y), "z": 0.0}
        bbox_max = {"x": float(v_max[0] + shift_x), "y": float(v_max[1] + shift_y), "z": float(v_max[2] + shift_z)}

        return pivot_offset, bbox_min, bbox_max

    def process_cleanup(
        self,
        job_id: str,
        raw_mesh_path: str,
        profile_type: CleanupProfileType = CleanupProfileType.MOBILE_DEFAULT,
        raw_texture_path: Optional[str] = None,
    ) -> Tuple[NormalizedMetadata, dict, str]:
        """
        Full cleanup orchestration.

        Returns:
            metadata, cleanup_stats, cleaned_mesh_path
        """
        if not os.path.exists(raw_mesh_path):
            raise FileNotFoundError(f"Raw mesh not found at: {raw_mesh_path}")

        # REQUIRED FIX: Auto-detect textured OBJ before any trimesh load or destructive path
        valid, reason, resolved_texture_path = self._is_valid_textured_obj_bundle(
            raw_mesh_path, raw_texture_path
        )
        logger.info("texture_safe_copy auto-detect: valid=%s reason=%s", valid, reason)

        if valid:
            logger.info("Found textured OBJ bundle: %s. Proceeding with object isolation check.", raw_mesh_path)
            # We don't return early here anymore. 
            # Instead, we will use the discovered texture later if we reach the texture safe copy path.
            pass

        profile = PROFILES[profile_type]

        job_cleaned_dir = self.cleaned_root / job_id
        job_cleaned_dir.mkdir(parents=True, exist_ok=True)

        if profile_type == CleanupProfileType.TEXTURE_SAFE_COPY:
            if not raw_texture_path:
                raise ValueError("TEXTURE_SAFE_COPY requires raw_texture_path")
            
            stats = self._run_texture_safe_copy(
                raw_mesh_path,
                raw_texture_path,
                job_cleaned_dir
            )
            
            # Generate metadata
            bbox_min = stats["bbox_min"]
            bbox_max = stats["bbox_max"]
            pivot_offset = stats["pivot_offset"]
            
            metadata = self.normalizer.generate_metadata(
                bbox_min,
                bbox_max,
                pivot_offset,
                stats["final_polycount"],
            )
            metadata_path = job_cleaned_dir / "normalized_metadata.json"
            self.normalizer.save_metadata(metadata, str(metadata_path))
            
            stats["metadata_path"] = str(metadata_path)
            stats["bbox_min"] = bbox_min
            stats["bbox_max"] = bbox_max
            
            return metadata, stats, stats["cleaned_mesh_path"]

        isolation_temp_path = job_cleaned_dir / "isolated_temp.obj"
        cleaned_mesh_path = job_cleaned_dir / "cleaned_mesh.obj"
        metadata_path = job_cleaned_dir / "normalized_metadata.json"

        # ------------------------------------------------------------
        # 1. Load raw mesh
        # ------------------------------------------------------------
        mesh = trimesh.load(raw_mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError(f"Raw mesh is empty or invalid: {raw_mesh_path}")

        # ------------------------------------------------------------
        # 2. Product isolation
        # ------------------------------------------------------------
        isolated_mesh, isolation_stats = self.isolator.isolate_product(mesh)

        if len(isolated_mesh.vertices) == 0 or len(isolated_mesh.faces) == 0:
            raise ValueError("Isolation produced an empty mesh")

        isolated_mesh.export(str(isolation_temp_path))

        pre_aligned_path = job_cleaned_dir / "pre_aligned_mesh.obj"

        # ------------------------------------------------------------
        # 3. Remesh / decimate
        # ------------------------------------------------------------
        final_polycount = self.remesher.process(
            str(isolation_temp_path),
            str(pre_aligned_path),
            profile,
        )

        # ------------------------------------------------------------
        # 4. Alignment
        # ------------------------------------------------------------
        _, pivot_offset = self.alignment.align_to_ground(
            str(pre_aligned_path),
            str(cleaned_mesh_path),
        )

        # ------------------------------------------------------------
        # 5. Bounding box
        # ------------------------------------------------------------
        bbox_min, bbox_max = self.bbox_extractor.extract(str(cleaned_mesh_path))

        # ------------------------------------------------------------
        # 6. Metadata
        # ------------------------------------------------------------
        metadata = self.normalizer.generate_metadata(
            bbox_min,
            bbox_max,
            pivot_offset,
            final_polycount,
        )
        self.normalizer.save_metadata(metadata, str(metadata_path))

        # ------------------------------------------------------------
        # 7. Temp cleanup
        # ------------------------------------------------------------
        if isolation_temp_path.exists():
            isolation_temp_path.unlink()

        # ------------------------------------------------------------
        # 8. Final artifact existence checks
        # ------------------------------------------------------------
        if not cleaned_mesh_path.exists():
            raise FileNotFoundError(f"Cleaned mesh artifact missing: {cleaned_mesh_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata artifact missing: {metadata_path}")

        # ------------------------------------------------------------
        # 9. Visual / texture inspection
        # ------------------------------------------------------------
        has_uv, has_material = self._inspect_visuals(str(cleaned_mesh_path))
        cleaned_texture_path = (
            str(raw_texture_path)
            if raw_texture_path and Path(raw_texture_path).exists()
            else None
        )

        # ------------------------------------------------------------
        # 10. Quality Gate (Item 4)
        # ------------------------------------------------------------
        quality_status = "success"
        quality_reason = "none"
        
        if final_polycount < 5000:
            quality_status = "quality_fail"
            quality_reason = f"Final polycount ({final_polycount}) is below the asset-quality threshold (5,000)."
        elif isolation_stats.get("component_count", 0) > 200:
            quality_status = "warning"
            quality_reason = "Large number of small islands detected. Mesh may be noisy."

        cleanup_stats = {
            "isolation": isolation_stats,
            "final_polycount": int(final_polycount),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "pre_aligned_mesh_path": str(pre_aligned_path),
            "cleaned_mesh_path": str(cleaned_mesh_path),
            "metadata_path": str(metadata_path),
            "cleaned_texture_path": cleaned_texture_path,
            "uv_preserved": bool(has_uv),
            "material_preserved": bool(has_material),
            "quality_status": quality_status,
            "quality_reason": quality_reason,
            "recapture_recommended": quality_status == "quality_fail"
        }

        return metadata, cleanup_stats, str(cleaned_mesh_path)