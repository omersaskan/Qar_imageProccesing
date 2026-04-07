from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class MeshMetadata(BaseModel):
    vertex_count: Optional[int] = None
    face_count: Optional[int] = None
    has_texture: bool = False
    bounding_box_min: Optional[Dict[str, float]] = None
    bounding_box_max: Optional[Dict[str, float]] = None

class OutputManifest(BaseModel):
    schema_version: int = Field(default=1)
    job_id: str
    mesh_path: str
    texture_path: Optional[str] = None
    log_path: str
    processing_time_seconds: float
    mesh_metadata: MeshMetadata = Field(default_factory=MeshMetadata)
    checksum: Optional[str] = None # SHA-256 of the mesh file
    engine_type: str = "unknown"
    is_stub: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
