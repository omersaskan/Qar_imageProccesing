from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class DeviceMetadata(BaseModel):
    platform: Optional[str] = None
    os: Optional[str] = None
    os_version: Optional[str] = None
    app_version: Optional[str] = None

class CaptureTrainingMetrics(BaseModel):
    duration_sec: float = 0.0
    resolution: Optional[str] = None
    fps: float = 0.0
    frame_count: int = 0
    bounding_box: Optional[List[float]] = None

class ReconstructionTrainingMetrics(BaseModel):
    vertex_count: int = 0
    face_count: int = 0
    density_metrics: Dict[str, Any] = Field(default_factory=dict)
    mesher_used: Optional[str] = None

class ExportTrainingMetrics(BaseModel):
    poly_count: int = 0
    texture_integrity_status: str = "missing"
    material_semantic_status: str = "geometry_only"

class TrainingLabels(BaseModel):
    asset_labels: List[str] = Field(default_factory=list)
    failure_reasons: List[str] = Field(default_factory=list)

class TrainingDataPaths(BaseModel):
    clean_model: Optional[str] = None
    original_video: Optional[str] = None
    frames_dir: Optional[str] = None

class TrainingDataManifest(BaseModel):
    schema_version: str = "1.0"
    session_id: str
    product_id_hash: str
    asset_id: Optional[str] = None
    created_at: str
    
    device: DeviceMetadata = Field(default_factory=DeviceMetadata)
    capture: CaptureTrainingMetrics = Field(default_factory=CaptureTrainingMetrics)
    reconstruction: ReconstructionTrainingMetrics = Field(default_factory=ReconstructionTrainingMetrics)
    export: ExportTrainingMetrics = Field(default_factory=ExportTrainingMetrics)
    labels: TrainingLabels = Field(default_factory=TrainingLabels)
    paths: TrainingDataPaths = Field(default_factory=TrainingDataPaths)
    
    consent_status: str = "unknown"
    eligible_for_training: bool = False
