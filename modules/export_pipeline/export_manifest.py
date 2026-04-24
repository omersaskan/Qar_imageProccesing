from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ExportArtifact(BaseModel):
    name: str
    artifact_type: str  # glb, usdz, poster, thumbnail
    file_path: str
    file_format: str
    file_size_bytes: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ExportManifest(BaseModel):
    job_id: str
    product_id: str
    artifacts: List[ExportArtifact] = Field(default_factory=list)
    export_profile: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
