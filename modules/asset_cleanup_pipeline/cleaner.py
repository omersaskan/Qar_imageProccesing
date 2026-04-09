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

        # ------------------------------------------------------------
        # 3. Remesh / decimate
        # ------------------------------------------------------------
        final_polycount = self.remesher.process(
            str(isolation_temp_path),
            str(cleaned_mesh_path),
            profile,
        )

        # ------------------------------------------------------------
        # 4. Alignment
        # ------------------------------------------------------------
        _, pivot_offset = self.alignment.align_to_ground(
            str(cleaned_mesh_path),
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
            "cleaned_mesh_path": str(cleaned_mesh_path),
            "metadata_path": str(metadata_path),
            "cleaned_texture_path": cleaned_texture_path,
            "uv_preserved": bool(has_uv),
            "material_preserved": bool(has_material),
        }

        return metadata, cleanup_stats, str(cleaned_mesh_path)