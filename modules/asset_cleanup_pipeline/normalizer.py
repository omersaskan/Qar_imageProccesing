import json
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, Field

class NormalizedMetadata(BaseModel):
    bbox_min: Dict[str, float]
    bbox_max: Dict[str, float]
    pivot_offset: Dict[str, float]
    final_polycount: int
    quality_score: float = 1.0 # 0.0 to 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Normalizer:
    def __init__(self):
        pass

    def generate_metadata(self, 
                          bbox_min: Dict[str, float], 
                          bbox_max: Dict[str, float], 
                          pivot_offset: Dict[str, float], 
                          polycount: int) -> NormalizedMetadata:
        """
        Creates the normalized metadata report.
        Real-world SCALE is excluded as requested; this belongs to the product profile.
        """
        return NormalizedMetadata(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            pivot_offset=pivot_offset,
            final_polycount=polycount
        )

    def save_metadata(self, metadata: NormalizedMetadata, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))
        
