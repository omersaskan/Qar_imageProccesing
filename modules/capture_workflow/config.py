from pydantic import BaseModel, Field


class QualityThresholds(BaseModel):
    # Blur detection (higher variance = sharper image)
    min_blur_score: float = Field(5.0, ge=0.0)

    # Exposure (0-255 scale)
    min_exposure_score: float = Field(40.0, ge=0.0, le=255.0)
    max_exposure_score: float = Field(230.0, ge=0.0, le=255.0)

    # Extraction & Similarity
    frame_sample_rate: int = Field(15, ge=1)  # Sample every Nth frame
    min_similarity_score: float = Field(0.95, ge=0.0, le=1.0)

    # Accepted object framing / purity
    min_object_occupancy: float = Field(0.05, ge=0.0, le=1.0)
    max_object_occupancy: float = Field(0.82, ge=0.0, le=1.0)
    max_center_distance: float = Field(0.42, ge=0.0)
    min_mask_confidence: float = Field(0.55, ge=0.0, le=1.0)
    min_mask_purity: float = Field(0.58, ge=0.0, le=1.0)
    max_mask_fragments: int = Field(3, ge=1)
    min_dominant_contour_ratio: float = Field(0.78, ge=0.0, le=1.0)
    min_mask_solidity: float = Field(0.55, ge=0.0, le=1.0)
    max_border_touch_ratio: float = Field(0.18, ge=0.0, le=1.0)
    max_bottom_band_ratio: float = Field(0.30, ge=0.0, le=1.0)
    max_bottom_span_ratio: float = Field(0.72, ge=0.0, le=1.0)
    max_support_area_ratio: float = Field(0.16, ge=0.0, le=1.0)
    support_alert_bottom_contact_ratio: float = Field(0.12, ge=0.0, le=1.0)
    support_alert_wide_span_ratio: float = Field(0.80, ge=0.0, le=1.0)
    support_alert_wide_band_ratio: float = Field(0.18, ge=0.0, le=1.0)

    # Mask generation
    seed_threshold: float = Field(0.42, ge=0.0, le=1.0)
    low_contrast_background_threshold: float = Field(0.12, ge=0.0, le=1.0)
    bottom_background_start_ratio: float = Field(0.88, ge=0.0, le=1.0)
    support_scan_start_ratio: float = Field(0.55, ge=0.0, le=1.0)
    support_min_span_ratio: float = Field(0.60, ge=0.0, le=1.0)
    support_max_height_ratio: float = Field(0.22, ge=0.0, le=1.0)
    support_min_area_ratio: float = Field(0.08, ge=0.0, le=1.0)
    support_min_y_ratio: float = Field(0.70, ge=0.0, le=1.0)
    support_min_remaining_ratio: float = Field(0.45, ge=0.0, le=1.0)
    confidence_target_occupancy: float = Field(0.22, ge=0.0, le=1.0)
    confidence_occupancy_tolerance: float = Field(0.30, gt=0.0, le=1.0)


class ExtractionConfig(BaseModel):
    max_frames: int = Field(100, ge=1)
    min_frames: int = Field(5, ge=1)

    # We intentionally suppress pixels outside the ROI rather than crop/resize.
    # Per-frame crop normalization would change effective intrinsics and can
    # destabilize photogrammetry engines like COLMAP.
    roi_mode: str = "mask_suppression"
    roi_pad_x_ratio: float = Field(0.18, ge=0.0, le=1.0)
    roi_pad_y_ratio: float = Field(0.20, ge=0.0, le=1.0)
    roi_min_retained_area_ratio: float = Field(0.65, ge=0.0, le=1.0)


class CoverageConfig(BaseModel):
    min_readable_frames: int = Field(8, ge=1)
    min_unique_views: int = Field(5, ge=1)
    dedupe_hist_similarity: float = Field(0.92, ge=0.0, le=1.0)
    dedupe_center_distance_max: float = Field(0.09, ge=0.0)
    dedupe_area_delta_max: float = Field(0.10, ge=0.0, le=1.0)
    dedupe_aspect_delta_max: float = Field(0.18, ge=0.0)
    dedupe_hu_delta_max: float = Field(0.45, ge=0.0)
    min_center_x_span: float = Field(0.10, ge=0.0)
    min_center_y_span: float = Field(0.08, ge=0.0)
    min_scale_variation: float = Field(1.15, ge=1.0)
    min_aspect_variation: float = Field(0.12, ge=0.0)
    elevated_view_scale_variation: float = Field(1.18, ge=1.0)
    elevated_view_aspect_variation: float = Field(0.14, ge=0.0)
    elevated_view_center_y_span: float = Field(0.10, ge=0.0)
    span_score_x_target: float = Field(0.18, gt=0.0)
    span_score_y_target: float = Field(0.14, gt=0.0)
    scale_score_target: float = Field(0.35, gt=0.0)
    aspect_score_target: float = Field(0.22, gt=0.0)


default_quality_thresholds = QualityThresholds()
default_extraction_config = ExtractionConfig()
default_coverage_config = CoverageConfig()
