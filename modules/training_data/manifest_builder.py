import hashlib
import json
from pathlib import Path
from typing import Optional
from .schema import TrainingDataManifest

class TrainingManifestBuilder:
    def __init__(self, data_root: Path):
        self.data_root = data_root

    def build(self, session_id: str, product_id: str, eligible_for_training: bool) -> TrainingDataManifest:
        # Hash product_id to anonymize
        product_hash = hashlib.sha256(product_id.encode()).hexdigest()[:16]
        
        manifest = TrainingDataManifest(
            session_id=session_id,
            product_hash=product_hash,
            eligible_for_training=eligible_for_training
        )
        
        session_dir = self.data_root / "sessions" / session_id
        if not session_dir.exists():
            return manifest
            
        # Try to enrich with validation report
        val_report_path = session_dir / "validation_report.json"
        if val_report_path.exists():
            try:
                with open(val_report_path, "r", encoding="utf-8") as f:
                    val_data = json.load(f)
                    
                manifest.poly_count = val_data.get("poly_count", 0)
                manifest.texture_integrity_status = val_data.get("texture_status", "missing")
                manifest.material_semantic_status = val_data.get("material_semantic_status", "geometry_only")
                manifest.contamination_score = val_data.get("contamination_score", 0.0)
            except Exception:
                pass
                
        # Basic paths
        asset_path = session_dir / "clean_model.glb"
        if asset_path.exists():
            try:
                manifest.asset_path = str(asset_path.relative_to(self.data_root))
            except ValueError:
                manifest.asset_path = str(asset_path)
                
        frames_dir = session_dir / "frames"
        if frames_dir.exists():
            try:
                manifest.original_frames_dir = str(frames_dir.relative_to(self.data_root))
            except ValueError:
                manifest.original_frames_dir = str(frames_dir)
                
        return manifest
