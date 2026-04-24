from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class TrainingDataManifest(BaseModel):
    session_id: str
    product_hash: str
    eligible_for_training: bool = False
    
    # QA Metrics
    poly_count: int = 0
    texture_integrity_status: str = "missing"
    material_semantic_status: str = "geometry_only"
    contamination_score: float = 0.0
    
    # Hardware/Environment (stripped of PII)
    platform: Optional[str] = None
    capture_resolution: Optional[str] = None
    
    # Taxonomy
    category_label: Optional[str] = None
    
    # Paths (relative to storage root)
    asset_path: Optional[str] = None
    original_frames_dir: Optional[str] = None
