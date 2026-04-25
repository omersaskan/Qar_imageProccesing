import os
from pathlib import Path
from typing import Tuple, Dict, Optional

import trimesh

from .profiles import PROFILES, CleanupProfileType
from .remesher import Remesher
from .alignment import AlignmentProcessor
from .bbox import BBoxExtractor
from .normalizer import Normalizer, NormalizedMetadata
from .isolation import MeshIsolator


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
            
        # Copy files
        cleaned_mesh_path = output_dir / "cleaned_mesh.obj"
        cleaned_mtl_path = output_dir / mtl_filename
        cleaned_tex_path = output_dir / raw_tex.name
        
        import shutil
        shutil.copy2(raw_mesh, cleaned_mesh_path)
        shutil.copy2(raw_tex, cleaned_tex_path)
        
        # Normalize MTL
        with open(cleaned_mtl_path, "w", encoding="utf-8") as f:
            f.write(f"newmtl material_0\n")
            f.write(f"Ka 1.000000 1.000000 1.000000\n")
            f.write(f"Kd 1.000000 1.000000 1.000000\n")
            f.write(f"Ks 0.000000 0.000000 0.000000\n")
            f.write(f"d 1.000000\n")
            f.write(f"illum 2\n")
            f.write(f"map_Kd {raw_tex.name}\n")
            
        # Get face count via trimesh but do NOT export/modify
        mesh = trimesh.load(str(cleaned_mesh_path))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        face_count = len(mesh.faces)
        
        return {
            "cleanup_mode": "texture_safe_copy",
            "uv_preserved": True,
            "material_preserved": True,
            "cleaned_mesh_path": str(cleaned_mesh_path),
            "cleaned_texture_path": str(cleaned_tex_path),
            "final_polycount": int(face_count),
        }

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
            bbox_min, bbox_max = self.bbox_extractor.extract(stats["cleaned_mesh_path"])
            metadata = self.normalizer.generate_metadata(
                bbox_min,
                bbox_max,
                {"x": 0.0, "y": 0.0, "z": 0.0}, # Texture safe copy doesn't realign yet
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
        }

        return metadata, cleanup_stats, str(cleaned_mesh_path)