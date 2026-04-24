from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, field_validator
from .lifecycle import AssetStatus, ReconstructionStatus

class Product(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SessionEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    from_status: str
    to_status: str
    note: Optional[str] = None
    stage: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CaptureSession(BaseModel):
    session_id: str
    product_id: str
    operator_id: str
    status: AssetStatus = AssetStatus.CREATED
    source_video_url: Optional[HttpUrl] = None
    source_video_path: Optional[str] = None
    extracted_frames: List[str] = Field(default_factory=list)
    coverage_score: float = Field(0.0, ge=0.0, le=1.0)
    failure_reason: Optional[str] = None
    reconstruction_job_id: Optional[str] = None
    reconstruction_manifest_path: Optional[str] = None
    cleanup_mesh_path: Optional[str] = None
    cleanup_metadata_path: Optional[str] = None
    cleanup_stats_path: Optional[str] = None
    export_blob_path: Optional[str] = None
    validation_report_path: Optional[str] = None
    asset_id: Optional[str] = None
    asset_version: Optional[str] = None
    publish_state: Optional[str] = None
    last_pipeline_stage: Optional[str] = None
    history: List[SessionEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # --- SPRINT 1: TICKET-001 — Retry Safety Fields ---
    # Number of recoverable-error retries attempted so far.
    retry_count: int = Field(0, ge=0)
    # The pipeline stage where the last retry occurred (e.g. "captured", "cleaned").
    last_retry_stage: Optional[str] = None
    # UTC timestamp of the most recent error/retry event.
    last_error_at: Optional[datetime] = None

    # --- SPRINT 2: TICKET-005 — Session-safe export metrics ---
    # Path to the persisted export_metrics JSON produced during _handle_validation.
    # This replaces the unsafe worker-instance-level _last_export_metrics cache that
    # could be overwritten by a concurrent session processed in the same worker cycle.
    export_metrics_path: Optional[str] = None

    # --- SPRINT 2: TICKET-006 — Progress-aware timeout ---
    # Updated whenever the session successfully advances a pipeline stage.
    # Timeout decisions use this timestamp rather than (only) created_at, so that
    # a long-but-progressing job is not incorrectly terminated.
    last_pipeline_progress_at: Optional[datetime] = None

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower()
        return v

class ReconstructionJobDraft(BaseModel):
    job_id: str
    capture_session_id: str
    input_frames: List[str]
    quality_report: Dict[str, Any] = Field(default_factory=dict)
    coverage_report: Dict[str, Any] = Field(default_factory=dict)
    product_id: str
    source_video_path: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ReconstructionJob(BaseModel):
    job_id: str
    capture_session_id: str
    product_id: str
    status: ReconstructionStatus = ReconstructionStatus.QUEUED
    input_frames: List[str]
    job_dir: str # Local path to job directory
    manifest_path: Optional[str] = None
    source_video_path: Optional[str] = None
    quality_report: Dict[str, Any] = Field(default_factory=dict)
    coverage_report: Dict[str, Any] = Field(default_factory=dict)
    failure_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class AssetMetadata(BaseModel):
    asset_id: str
    product_id: str
    version: Optional[str] = None
    bbox: Dict[str, Any] = Field(default_factory=dict)
    pivot_offset: Dict[str, Any] = Field(default_factory=dict)
    quality_grade: str = "C" # A, B, C, D
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ProductPhysicalProfile(BaseModel):
    real_width_cm: float = Field(..., gt=0)
    real_depth_cm: float = Field(..., gt=0)
    real_height_cm: float = Field(..., gt=0)
    plate_diameter_cm: Optional[float] = Field(None, gt=0)
    ground_offset_cm: float = 0.0
    recommended_scale_multiplier: float = Field(1.0, gt=0)

class ValidationReport(BaseModel):
    asset_id: str
    poly_count: int = Field(..., ge=0)
    texture_status: str
    bbox_reasonable: bool
    ground_aligned: bool
    mobile_performance_grade: str

    # Contamination metrics
    component_count: int = 1
    largest_component_share: float = 1.0
    contamination_score: float = 0.0 # 0.0 (clean) to 1.0 (contaminated)
    contamination_report: Dict[str, Any] = Field(default_factory=dict)

    final_decision: str # pass, fail, review

    # Phase 3: New Robust Scoring
    flatness_score: float = 1.0
    compactness_score: float = 1.0
    selected_component_score: float = 1.0

    # Phase 2.2C: Material Semantics
    material_quality_grade: str = "F"
    material_semantic_status: str = "geometry_only"

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AssetPackage(BaseModel):
    product_id: str
    asset_version: str
    glb_url: HttpUrl
    usdz_url: HttpUrl
    poster_image_url: HttpUrl
    thumbnail_url: HttpUrl
    bbox: Dict[str, Any]
    pivot_offset: Dict[str, Any]
    physical_profile: ProductPhysicalProfile
    validation_status: str
    package_status: str = "ready_for_ar"

class GuidanceSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class CaptureGuidance(BaseModel):
    session_id: str
    status: AssetStatus
    next_action: str
    should_recapture: bool
    is_ready_for_review: bool
    messages: List[Dict[str, Any]] = Field(default_factory=list) # [{code: str, message: str, severity: GuidanceSeverity}]
    coverage_summary: Optional[Dict[str, Any]] = None
    validation_summary: Optional[Dict[str, Any]] = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# --- ADAPTIVE RECONSTRUCTION FALLBACK MODELS ---

class ReconstructionAttemptType(str, Enum):
    DEFAULT = "default"
    DENSER_FRAMES = "denser_frames"
    UNMASKED = "unmasked"

class ReconstructionAttemptResult(BaseModel):
    attempt_type: ReconstructionAttemptType
    status: str # "success", "weak", "failed"
    frames_used: int
    registered_images: int = 0
    sparse_points: int = 0
    dense_points_fused: int = 0
    mesher_used: str = "none"
    mesh_path: Optional[str] = None
    log_path: Optional[str] = None
    error_message: Optional[str] = None
    
    # Re-extraction evidence
    sampling_rate_used: Optional[int] = None
    source_video_path: Optional[str] = None
    reextracted_frames_dir: Optional[str] = None
    
    metrics_rank_score: float = 0.0 # internal score used for selection
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ReconstructionAudit(BaseModel):
    capture_session_id: str
    attempts: List[ReconstructionAttemptResult] = Field(default_factory=list)
    selected_best_index: Optional[int] = None
    final_status: str = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
