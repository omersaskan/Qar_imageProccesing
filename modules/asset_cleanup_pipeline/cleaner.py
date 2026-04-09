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
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        has_uv = False
        has_material = False

        try:
            has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
        except Exception:
            has_uv = False

        try:
            has_material = hasattr(mesh.visual, "material") and mesh.visual.material is not None
        except Exception:
            has_material = False

        return bool(has_uv), bool(has_material)

    def process_cleanup(
        self,
        job_id: str,
        raw_mesh_path: str,
        profile_type: CleanupProfileType = CleanupProfileType.MOBILE_DEFAULT,
        raw_texture_path: Optional[str] = None,
    ) -> Tuple[NormalizedMetadata, dict, str]:
        if not os.path.exists(raw_mesh_path):
            raise FileNotFoundError(f"Raw mesh not found at: {raw_mesh_path}")

        profile = PROFILES[profile_type]
        job_cleaned_dir = self.cleaned_root / job_id
        job_cleaned_dir.mkdir(parents=True, exist_ok=True)

        isolation_temp_path = job_cleaned_dir / "isolated_temp.obj"
        cleaned_mesh_path = job_cleaned_dir / "cleaned_mesh.obj"
        metadata_path = job_cleaned_dir / "normalized_metadata.json"

        mesh = trimesh.load(raw_mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        isolated_mesh, isolation_stats = self.isolator.isolate_product(mesh)
        isolated_mesh.export(str(isolation_temp_path))

        final_polycount = self.remesher.process(str(isolation_temp_path), str(cleaned_mesh_path), profile)
        _, pivot_offset = self.alignment.align_to_ground(str(cleaned_mesh_path), str(cleaned_mesh_path))

        bbox_min, bbox_max = self.bbox_extractor.extract(str(cleaned_mesh_path))
        metadata = self.normalizer.generate_metadata(bbox_min, bbox_max, pivot_offset, final_polycount)
        self.normalizer.save_metadata(metadata, str(metadata_path))

        if isolation_temp_path.exists():
            isolation_temp_path.unlink()

        if not cleaned_mesh_path.exists():
            raise FileNotFoundError(f"Cleaned mesh artifact not found at: {cleaned_mesh_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata artifact not found at: {metadata_path}")

        has_uv, has_material = self._inspect_visuals(str(cleaned_mesh_path))
        cleaned_texture_path = raw_texture_path if raw_texture_path and os.path.exists(raw_texture_path) else None

        cleanup_stats = {
            "isolation": isolation_stats,
            "final_polycount": final_polycount,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "cleaned_mesh_path": str(cleaned_mesh_path),
            "metadata_path": str(metadata_path),
            "cleaned_texture_path": cleaned_texture_path,
            "uv_preserved": has_uv,
            "material_preserved": has_material,
        }

        return metadata, cleanup_stats, str(cleaned_mesh_path)