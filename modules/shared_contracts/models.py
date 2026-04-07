from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, field_validator
from .lifecycle import AssetStatus, ReconstructionStatus

class Product(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CaptureSession(BaseModel):
    session_id: str
    product_id: str
    operator_id: str
    status: AssetStatus = AssetStatus.CREATED
    source_video_url: Optional[HttpUrl] = None
    extracted_frames: List[HttpUrl] = Field(default_factory=list)
    coverage_score: float = Field(0.0, ge=0.0, le=1.0)
    failure_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ReconstructionJob(BaseModel):
    job_id: str
    capture_session_id: str
    product_id: str
    status: ReconstructionStatus = ReconstructionStatus.QUEUED
    input_frames: List[str]
    job_dir: str # Local path to job directory
    manifest_path: Optional[str] = None
    failure_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class AssetMetadata(BaseModel):
    asset_id: str
    product_id: str
    version: str
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
    final_decision: str # pass, fail, review
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
