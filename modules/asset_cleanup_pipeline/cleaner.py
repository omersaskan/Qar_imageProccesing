import os
from pathlib import Path
from typing import Optional, Tuple, Dict
from .profiles import CleanupProfile, PROFILES, CleanupProfileType
from .remesher import Remesher
from .alignment import AlignmentProcessor
from .bbox import BBoxExtractor
from .normalizer import Normalizer, NormalizedMetadata
from modules.shared_contracts.errors import MetadataCorruptionError, PathSafetyError
from modules.utils.path_safety import validate_safe_path, ensure_dir

from .isolation import MeshIsolator
import trimesh

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

    def process_cleanup(self, 
                        job_id: str, 
                        raw_mesh_path: str, 
                        profile_type: CleanupProfileType = CleanupProfileType.MOBILE_DEFAULT) -> Tuple[NormalizedMetadata, dict, str]:
        """
        Orchestrates the full product-centric cleanup pipeline.
        Returns (metadata, cleanup_stats).
        """
        if not os.path.exists(raw_mesh_path):
            raise FileNotFoundError(f"Raw mesh not found at: {raw_mesh_path}")

        profile = PROFILES[profile_type]
        job_cleaned_dir = self.cleaned_root / job_id
        job_cleaned_dir.mkdir(parents=True, exist_ok=True)
        
        isolation_temp_path = job_cleaned_dir / "isolated_temp.obj"
        cleaned_mesh_path = job_cleaned_dir / "cleaned_mesh.obj"
        metadata_path = job_cleaned_dir / "normalized_metadata.json"

        # 1. Isolate Product (Remove planes, noise)
        mesh = trimesh.load(raw_mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
            
        isolated_mesh, isolation_stats = self.isolator.isolate_product(mesh)
        isolated_mesh.export(str(isolation_temp_path))

        # 2. Remesh / Decimate
        final_polycount = self.remesher.process(str(isolation_temp_path), str(cleaned_mesh_path), profile)
        
        # 3. Alignment & Pivot (inplace on cleaned_mesh)
        _, pivot_offset = self.alignment.align_to_ground(str(cleaned_mesh_path), str(cleaned_mesh_path))
        
        # 4. Bounding Box extraction
        bbox_min, bbox_max = self.bbox_extractor.extract(str(cleaned_mesh_path))

        # 5. Generate & Save Normalized Metadata 
        metadata = self.normalizer.generate_metadata(bbox_min, bbox_max, pivot_offset, final_polycount)
        self.normalizer.save_metadata(metadata, str(metadata_path))
        
        # Phase 1: Check texture preservation
        # Reload mesh to check visuals after all processing
        final_mesh = trimesh.load(str(cleaned_mesh_path))
        uv_preserved = hasattr(final_mesh.visual, 'uv') and final_mesh.visual.uv is not None
        material_preserved = hasattr(final_mesh.visual, 'material') and final_mesh.visual.material is not None

        # Cleanup temp
        if isolation_temp_path.exists():
            isolation_temp_path.unlink()

        # 6. Final verification of artifacts
        if not cleaned_mesh_path.exists():
            raise FileNotFoundError(f"Cleaned mesh artifact not found at: {cleaned_mesh_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata artifact not found at: {metadata_path}")
            
        # Combine stats
        cleanup_stats = {
            "isolation": isolation_stats,
            "final_polycount": final_polycount,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "cleaned_mesh_path": str(cleaned_mesh_path),
            "metadata_path": str(metadata_path),
            "uv_preserved": uv_preserved,
            "material_preserved": material_preserved
        }

        return metadata, cleanup_stats, str(cleaned_mesh_path)
