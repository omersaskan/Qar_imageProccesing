import os
from pathlib import Path
from typing import Optional
from .profiles import CleanupProfile, PROFILES, CleanupProfileType
from .remesher import Remesher
from .alignment import AlignmentProcessor
from .bbox import BBoxExtractor
from .normalizer import Normalizer, NormalizedMetadata
from modules.shared_contracts.errors import MetadataCorruptionError, PathSafetyError
from modules.utils.path_safety import validate_safe_path, ensure_dir

class AssetCleaner:
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.cleaned_root = self.data_root / "cleaned"
        self.cleaned_root.mkdir(parents=True, exist_ok=True)
        
        self.remesher = Remesher()
        self.alignment = AlignmentProcessor()
        self.bbox_extractor = BBoxExtractor()
        self.normalizer = Normalizer()

    def process_cleanup(self, 
                        job_id: str, 
                        raw_mesh_path: str, 
                        profile_type: CleanupProfileType = CleanupProfileType.MOBILE_DEFAULT) -> NormalizedMetadata:
        """
        Orchestrates the full cleanup pipeline.
        """
        if not os.path.exists(raw_mesh_path):
            raise FileNotFoundError(f"Raw mesh not found at: {raw_mesh_path}")

        profile = PROFILES[profile_type]
        job_cleaned_dir = self.cleaned_root / job_id
        job_cleaned_dir.mkdir(parents=True, exist_ok=True)
        
        cleaned_mesh_path = job_cleaned_dir / "cleaned_mesh.obj"
        metadata_path = job_cleaned_dir / "normalized_metadata.json"

        # 1. Remesh / Decimate
        final_polycount = self.remesher.process(raw_mesh_path, str(cleaned_mesh_path), profile)
        
        # 2. Alignment & Pivot (inplace on cleaned_mesh or returns new path)
        _, pivot_offset = self.alignment.align_to_ground(str(cleaned_mesh_path), str(cleaned_mesh_path))
        
        # 3. Bounding Box extraction
        bbox_min, bbox_max = self.bbox_extractor.extract(str(cleaned_mesh_path))

        # 4. Generate & Save Normalized Metadata 
        metadata = self.normalizer.generate_metadata(bbox_min, bbox_max, pivot_offset, final_polycount)
        self.normalizer.save_metadata(metadata, str(metadata_path))
        
        # 5. Finalize Artifact Existence Checks
        if not os.path.exists(str(cleaned_mesh_path)):
            raise FileNotFoundError(f"Cleanup failed: Cleaned mesh missing for job {job_id}")
            
        if not os.path.exists(str(metadata_path)):
             raise MetadataCorruptionError(f"Cleanup failed: Normalized metadata missing for job {job_id}")

        return metadata
