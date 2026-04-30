from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class MeshMetadata(BaseModel):
    vertex_count: Optional[int] = None
    face_count: Optional[int] = None
    has_texture: bool = False
    uv_present: bool = False
    bounding_box_min: Optional[Dict[str, float]] = None
    bounding_box_max: Optional[Dict[str, float]] = None
    texture_resolution: Optional[str] = None

class OutputManifest(BaseModel):
    schema_version: int = Field(default=2)
    job_id: str
    mesh_path: str
    textured_mesh_path: Optional[str] = None
    texture_path: Optional[str] = None # Legacy fallback or primary map
    texture_atlas_paths: List[str] = Field(default_factory=list)
    log_path: str
    texturing_log_path: Optional[str] = None
    processing_time_seconds: float
    mesh_metadata: MeshMetadata = Field(default_factory=MeshMetadata)
    checksum: Optional[str] = None # SHA-256 of the mesh file
    engine_type: str = "unknown"
    texturing_engine: Optional[str] = None
    texturing_status: str = "absent" # "absent", "real", "degraded"
    is_stub: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # --- Phase B: Safety & Generative Info ---
    ai_generated: bool = False
    geometry_source: str = "photogrammetry"
    production_status: str = "production_candidate"
    requires_manual_review: bool = False
    may_override_recapture_required: bool = False
