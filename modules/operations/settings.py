from enum import Enum
from typing import Optional, List
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class AppEnvironment(str, Enum):
    LOCAL_DEV = "local_dev"
    PILOT = "pilot"
    PRODUCTION = "production"

class ReconstructionPipeline(str, Enum):
    COLMAP_DENSE = "colmap_dense"
    COLMAP_OPENMVS = "colmap_openmvs"
    SIMULATED = "simulated"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Operational
    env: AppEnvironment = Field(AppEnvironment.LOCAL_DEV, validation_alias="ENV")
    data_root: str = Field("data", validation_alias="DATA_ROOT")
    worker_interval_sec: int = Field(5, validation_alias="WORKER_INTERVAL_SEC")
    pilot_api_key: Optional[str] = Field(None, validation_alias="PILOT_API_KEY")
    embedded_worker_enabled: bool = Field(True, validation_alias="MESHYSIZ_EMBEDDED_WORKER")

    # Binaries
    colmap_path: str = Field(default_factory=lambda: __import__("shutil").which("colmap") or "colmap", validation_alias="RECON_ENGINE_PATH")
    openmvs_path: str = Field(
        default_factory=lambda: __import__("os").environ.get("OPENMVS_BIN_PATH") or __import__("os").environ.get("OPENMVS_BIN") or "/usr/local/bin", 
        validation_alias="OPENMVS_BIN_PATH"
    )
    ffmpeg_path: str = Field(
        default_factory=lambda: __import__("shutil").which("ffmpeg") or "ffmpeg", 
        validation_alias="FFMPEG_PATH"
    )
    ffprobe_path: Optional[str] = Field(
        default_factory=lambda: __import__("shutil").which("ffprobe") or "ffprobe",
        validation_alias="FFPROBE_PATH"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # SPRINT: OpenMVS env harmonization
        # OPENMVS_BIN_PATH wins, but we fallback to OPENMVS_BIN if set
        import os
        if not os.environ.get("OPENMVS_BIN_PATH") and os.environ.get("OPENMVS_BIN"):
            self.openmvs_path = os.environ.get("OPENMVS_BIN")
        
        # Derive ffprobe from ffmpeg if not set
        if not self.ffprobe_path and self.ffmpeg_path:
            fp = Path(self.ffmpeg_path)
            if fp.name == "ffmpeg.exe":
                probe_p = fp.parent / "ffprobe.exe"
                if probe_p.exists():
                    self.ffprobe_path = str(probe_p)
            elif fp.name == "ffmpeg":
                probe_p = fp.parent / "ffprobe"
                if probe_p.exists():
                    self.ffprobe_path = str(probe_p)
    use_gpu: bool = Field(True, validation_alias="RECON_USE_GPU")
    gpu_index: str = Field("0", validation_alias="RECON_GPU_INDEX")
    recon_max_image_size: int = Field(2000, validation_alias="RECON_MAX_IMAGE_SIZE")
    recon_matcher: str = Field("exhaustive", validation_alias="RECON_MATCHER")

    # Retention
    published_frames_days: int = Field(3, validation_alias="RETENTION_PUBLISHED_FRAMES_DAYS")
    failed_frames_days: int = Field(14, validation_alias="RETENTION_FAILED_FRAMES_DAYS")
    reconstruction_scratch_hours: int = Field(48, validation_alias="RETENTION_RECON_SCRATCH_HOURS")
    # SPRINT 3 TICKET-011: Draft sessions (validated but not yet published/failed)
    # retain raw frames longer than published assets but shorter than failed ones.
    draft_frames_days: int = Field(7, validation_alias="RETENTION_DRAFT_FRAMES_DAYS")


    # Enforcement
    strict_ml_segmentation: bool = Field(True, validation_alias="STRICT_ML_SEGMENTATION")

    # --- SPRINT 1: TICKET-001 — Retry & Timeout Safety ---
    # Maximum number of recoverable retries per session before the session is
    # forcibly moved to FAILED. Prevents infinite retry loops. Default: 5.
    max_retry_count: int = Field(5, validation_alias="MAX_RETRY_COUNT")
    # Sessions stuck in a processing state longer than this are forcibly FAILED.
    # (in hours). Default: 2h.
    session_timeout_hours: int = Field(2, validation_alias="SESSION_TIMEOUT_HOURS")

    # --- ADAPTIVE RECONSTRUCTION FALLBACK ---
    # Order of attempts to try if reconstruction is weak.
    # Options: "default", "denser_frames", "unmasked"
    recon_fallback_steps: list[str] = Field(["default", "denser_frames"], validation_alias="RECON_FALLBACK_STEPS")
    # Whether unmasked fallback is even allowed.
    recon_unmasked_fallback_enabled: bool = Field(False, validation_alias="RECON_UNMASKED_FALLBACK_ENABLED")
    
    # SPRINT 4: Diagnostic toggle to force-enable unmasked fallback for troubleshooting.
    recon_diagnostic_enable_unmasked: bool = Field(False, validation_alias="RECON_DIAGNOSTIC_ENABLE_UNMASKED")
    
    # SPRINT 5: Hybrid masking (unmasked pose, masked object)
    recon_hybrid_masking: bool = Field(True, validation_alias="RECON_HYBRID_MASKING")
    
    # Default effort will use whatever is on disk (usually from extraction rate=15).
    # Fallback DENSER_FRAMES will re-extract with this rate.
    recon_fallback_sample_rate: int = Field(5, validation_alias="RECON_FALLBACK_SAMPLE_RATE")

    # For denser_frames: 1.0 means use all frames, lower means more sparse. 
    recon_denser_sampling_ratio: float = Field(1.0, validation_alias="RECON_DENSER_SAMPLING_RATIO")

    # --- SPRINT 1: TICKET-004 — Disk Space Preflight ---
    # Minimum free disk space on the data_root partition before an upload is
    # accepted. Stricter environments should raise this. Default: 5 GB.
    min_free_disk_gb: float = Field(5.0, validation_alias="MIN_FREE_DISK_GB")

    # Upload Preflight Strict Defaults
    max_upload_mb: float = Field(500.0, validation_alias="MAX_UPLOAD_MB")
    min_video_duration_sec: float = Field(15.0, validation_alias="MIN_VIDEO_DURATION_SEC")
    max_video_duration_sec: float = Field(120.0, validation_alias="MAX_VIDEO_DURATION_SEC")
    min_video_fps: float = Field(20.0, validation_alias="MIN_VIDEO_FPS")
    min_video_short_edge: int = Field(720, validation_alias="MIN_VIDEO_SHORT_EDGE")
    min_video_long_edge: int = Field(1280, validation_alias="MIN_VIDEO_LONG_EDGE")
    min_video_width: int = Field(720, validation_alias="MIN_VIDEO_WIDTH")
    min_video_height: int = Field(720, validation_alias="MIN_VIDEO_HEIGHT")
    
    video_probe_timeout_sec: int = Field(30, validation_alias="VIDEO_PROBE_TIMEOUT_SEC")
    video_normalize_timeout_sec: int = Field(180, validation_alias="VIDEO_NORMALIZE_TIMEOUT_SEC")
    
    # --- SECURITY: CORS Allowlist ---
    # List of origins allowed to access the API. 
    # In LOCAL_DEV, it defaults to ["*"] if not set.
    # In Pilot/Production, it MUST be explicitly configured.
    cors_allow_origins: List[str] = Field(default_factory=list, validation_alias="CORS_ALLOW_ORIGINS")

    def get_cors_origins(self) -> List[str]:
        if self.cors_allow_origins:
            return self.cors_allow_origins
        if self.env == AppEnvironment.LOCAL_DEV:
            return ["*"]
        return [] # Secure default for production

    # --- AR CAPTURE QUALITY GATING ---
    ar_min_coverage: float = Field(90.0, validation_alias="AR_MIN_COVERAGE")
    ar_max_gap: float = Field(45.0, validation_alias="AR_MAX_GAP")
    ar_min_accepted_frames: int = Field(100, validation_alias="AR_MIN_ACCEPTED_FRAMES")
    ar_max_blur_ratio: float = Field(0.3, validation_alias="AR_MAX_BLUR_RATIO")
    ar_min_duration_sec: float = Field(15.0, validation_alias="AR_MIN_DURATION_SEC")
    ar_manifest_frame_count_tolerance: float = Field(0.20, validation_alias="AR_MANIFEST_FRAME_COUNT_TOLERANCE")
    ar_min_accepted_ratio: float = Field(0.35, validation_alias="AR_MIN_ACCEPTED_RATIO")

    # --- RECONSTRUCTION ENGINES ---
    # Global switch for the default pipeline: "colmap_dense" or "colmap_openmvs"
    recon_pipeline: str = Field("colmap_dense", validation_alias="RECON_PIPELINE")
    
    # Poisson mesher timeout in seconds.
    recon_poisson_timeout_sec: int = Field(300, validation_alias="RECON_POISSON_TIMEOUT_SEC")
    
    # COLMAP StereoFusion Thresholds
    recon_stereo_fusion_min_num_pixels: int = Field(2, validation_alias="RECON_STEREO_FUSION_MIN_NUM_PIXELS")
    recon_stereo_fusion_max_reproj_error: float = Field(2.0, validation_alias="RECON_STEREO_FUSION_MAX_REPROJ_ERROR")
    recon_stereo_fusion_max_depth_error: float = Field(0.01, validation_alias="RECON_STEREO_FUSION_MAX_DEPTH_ERROR")
    recon_stereo_fusion_max_normal_error: float = Field(10.0, validation_alias="RECON_STEREO_FUSION_MAX_NORMAL_ERROR")

    # Poisson Parameters
    recon_poisson_depth: int = Field(10, validation_alias="RECON_POISSON_DEPTH")
    recon_poisson_trim: int = Field(7, validation_alias="RECON_POISSON_TRIM")
    
    # Mesh Budget and Decimation Gates
    # Max faces allowed for direct object isolation (Part 3). If exceeded, use pre-decimation.
    recon_mesh_budget_faces: int = Field(2_000_000, validation_alias="RECON_MESH_BUDGET_FACES")
    # Hard limit. If even after pre-decimation it's still too large, fail fast.
    recon_mesh_hard_limit_faces: int = Field(8_000_000, validation_alias="RECON_MESH_HARD_LIMIT_FACES")
    
    # Decimation targets
    recon_pre_cleanup_target_faces: int = Field(800_000, validation_alias="RECON_PRE_CLEANUP_TARGET_FACES")
    recon_mobile_target_faces: int = Field(100_000, validation_alias="RECON_MOBILE_TARGET_FACES")

    # Poisson retry settings
    recon_poisson_depth_retry: int = Field(9, validation_alias="RECON_POISSON_DEPTH_RETRY")
    recon_poisson_trim_retry: int = Field(8, validation_alias="RECON_POISSON_TRIM_RETRY")
    
    # Python load limit (safe threshold for trimesh.load)
    max_faces_python_decimation: int = Field(1_500_000, validation_alias="MAX_FACES_PYTHON_DECIMATION")

    # OpenMVS specific flags
    openmvs_fail_hard: bool = Field(False, validation_alias="OPENMVS_FAIL_HARD")
    openmvs_textured_output: bool = Field(True, validation_alias="OPENMVS_TEXTURED_OUTPUT")
    require_textured_output: bool = Field(False, validation_alias="REQUIRE_TEXTURED_OUTPUT")
    fail_on_texture_missing: bool = Field(True, validation_alias="FAIL_ON_TEXTURE_MISSING")
    fail_on_uv_missing: bool = Field(True, validation_alias="FAIL_ON_UV_MISSING")
    min_texture_resolution: int = Field(1024, validation_alias="MIN_TEXTURE_RESOLUTION")
    max_empty_texture_ratio: float = Field(0.2, validation_alias="MAX_EMPTY_TEXTURE_RATIO")

    # --- TEXTURE MASK REFINEMENT ---
    texture_mask_erode_px: int = Field(3, validation_alias="TEXTURE_MASK_ERODE_PX")
    texture_reject_support_contamination: bool = Field(True, validation_alias="TEXTURE_REJECT_SUPPORT_CONTAMINATION")
    texture_reject_subject_clipped: bool = Field(True, validation_alias="TEXTURE_REJECT_SUBJECT_CLIPPED")
    texture_min_clean_frames: int = Field(20, validation_alias="TEXTURE_MIN_CLEAN_FRAMES")

    # --- TEXTURE MESH SIMPLIFICATION & RETRY ---
    texture_texturing_target_faces: int = Field(60000, validation_alias="TEXTURE_TEXTURING_TARGET_FACES")
    texture_safe_texturing_target_faces: int = Field(40000, validation_alias="TEXTURE_SAFE_TEXTURING_TARGET_FACES")
    texture_native_crash_retry_faces: int = Field(30000, validation_alias="TEXTURE_NATIVE_CRASH_RETRY_FACES")
    texture_use_compatible_neutralization: bool = Field(True, validation_alias="TEXTURE_USE_COMPATIBLE_NEUTRALIZATION")
    texture_retry_raw_all: bool = Field(True, validation_alias="TEXTURE_RETRY_RAW_ALL")
    texture_max_selected_frames: int = Field(20, validation_alias="TEXTURE_MAX_SELECTED_FRAMES")
    texture_neutralization_type: str = Field("cream", validation_alias="TEXTURE_NEUTRALIZATION_TYPE")
    texture_timeout_sec: int = Field(600, validation_alias="TEXTURE_TIMEOUT_SEC")

    # --- TEXTURE QUALITY QA ---
    max_black_pixel_ratio: float = Field(0.20, validation_alias="MAX_BLACK_PIXEL_RATIO")
    max_dominant_background_ratio: float = Field(0.15, validation_alias="MAX_DOMINANT_BACKGROUND_RATIO")
    min_atlas_coverage_ratio: float = Field(0.60, validation_alias="MIN_ATLAS_COVERAGE_RATIO")
    min_near_white_ratio_white_cream: float = Field(0.40, validation_alias="MIN_NEAR_WHITE_RATIO_WHITE_CREAM")
    white_cream_max_background_ratio: Optional[float] = Field(None, validation_alias="WHITE_CREAM_MAX_BACKGROUND_RATIO")
    expected_product_color: str = Field("unknown", validation_alias="EXPECTED_PRODUCT_COLOR")
    
    # --- TEXTURE NORMALIZATION (SPRINT 6) ---
    texture_brightness_target: float = Field(240.0, validation_alias="TEXTURE_BRIGHTNESS_TARGET")
    texture_gamma: float = Field(0.85, validation_alias="TEXTURE_GAMMA")
    texture_saturation: float = Field(1.1, validation_alias="TEXTURE_SATURATION")
    texture_roughness: float = Field(0.8, validation_alias="TEXTURE_ROUGHNESS")
    texture_metallic: float = Field(0.0, validation_alias="TEXTURE_METALLIC")
    texture_unlit: bool = Field(True, validation_alias="TEXTURE_UNLIT")

    # --- CAPTURE PROFILE (size_class × scene_type aware pipeline tuning) ---
    # Options: small_on_surface, small_freestanding, small_mounted,
    #          medium_on_surface, medium_freestanding, medium_mounted,
    #          large_on_surface, large_freestanding, large_mounted
    capture_profile: str = Field("small_on_surface", validation_alias="CAPTURE_PROFILE")
    material_hint: str = Field("opaque", validation_alias="MATERIAL_HINT")

    # --- SPRINT 3: Adaptive Keyframe Sampling ---
    # When false, frame_extractor uses the legacy fixed `frame_sample_rate`.
    # When true, AdaptiveSampler decides per-frame keep/skip via optical flow.
    adaptive_sampling_enabled: bool = Field(False, validation_alias="ADAPTIVE_SAMPLING_ENABLED")
    coverage_aware_rebalance_enabled: bool = Field(False, validation_alias="COVERAGE_AWARE_REBALANCE_ENABLED")

    # --- SPRINT 4: Reconstruction Preset Hardening ---
    # When false, runner uses the existing global settings tuning verbatim.
    # When true, reconstruction_profile + preset_resolver pick safer per-job presets.
    reconstruction_preset_hardening_enabled: bool = Field(
        False, validation_alias="RECONSTRUCTION_PRESET_HARDENING_ENABLED"
    )
    intrinsics_cache_enabled: bool = Field(False, validation_alias="INTRINSICS_CACHE_ENABLED")
    intrinsics_feed_to_colmap_enabled: bool = Field(
        False, validation_alias="INTRINSICS_FEED_TO_COLMAP_ENABLED"
    )
    # Sprint 4.5: how many fallback attempts to try when an attempt fails.
    # Hard cap to prevent infinite retry loops. 0 = no retry.
    fallback_ladder_max_attempts: int = Field(
        3, validation_alias="FALLBACK_LADDER_MAX_ATTEMPTS"
    )
    # Sprint 4.6: when true (and hardening enabled), runner.run() drives
    # attempts via the preset-aware fallback ladder instead of the legacy
    # recon_fallback_steps loop.  Default false preserves Sprint 4.5
    # manifest-only behaviour.
    reconstruction_runtime_fallback_enabled: bool = Field(
        False, validation_alias="RECONSTRUCTION_RUNTIME_FALLBACK_ENABLED"
    )
    # Sprint 5: pose-backed coverage matrix from COLMAP sparse output.
    pose_backed_coverage_enabled: bool = Field(
        False, validation_alias="POSE_BACKED_COVERAGE_ENABLED"
    )
    # Sprint 6: Blender headless cleanup + GLB re-export (opt-in).
    blender_cleanup_enabled: bool = Field(
        False, validation_alias="BLENDER_CLEANUP_ENABLED"
    )
    blender_cleanup_decimate_enabled: bool = Field(
        False, validation_alias="BLENDER_CLEANUP_DECIMATE_ENABLED"
    )
    blender_cleanup_decimate_ratio: float = Field(
        0.5, validation_alias="BLENDER_CLEANUP_DECIMATE_RATIO"
    )
    # Sprint 7: glTF-Transform optimization + Khronos validation gate (opt-in).
    gltf_optimization_enabled: bool = Field(
        False, validation_alias="GLTF_OPTIMIZATION_ENABLED"
    )
    gltf_validation_enabled: bool = Field(
        False, validation_alias="GLTF_VALIDATION_ENABLED"
    )
    gltf_validation_reject_on_error: bool = Field(
        True, validation_alias="GLTF_VALIDATION_REJECT_ON_ERROR"
    )
    # Sprint 8: license manifest + provenance (opt-in).
    license_manifest_enabled: bool = Field(
        False, validation_alias="LICENSE_MANIFEST_ENABLED"
    )
    provenance_enabled: bool = Field(
        False, validation_alias="PROVENANCE_ENABLED"
    )

    # --- PHASE 3B: DEPTH STUDIO ---
    single_image_depth_enabled: bool = Field(False, validation_alias="SINGLE_IMAGE_DEPTH_ENABLED")
    depth_studio_enabled: bool = Field(False, validation_alias="DEPTH_STUDIO_ENABLED")
    depth_studio_allow_video_input: bool = Field(True, validation_alias="DEPTH_STUDIO_ALLOW_VIDEO_INPUT")
    depth_studio_default_provider: str = Field("depth_anything_v2", validation_alias="DEPTH_STUDIO_DEFAULT_PROVIDER")
    depth_pro_enabled: bool = Field(False, validation_alias="DEPTH_PRO_ENABLED")
    depth_pro_python_path: Optional[str] = Field(None, validation_alias="DEPTH_PRO_PYTHON_PATH")
    depth_output_format: str = Field("png16", validation_alias="DEPTH_OUTPUT_FORMAT")
    depth_output_allow_exr: bool = Field(True, validation_alias="DEPTH_OUTPUT_ALLOW_EXR")
    depth_mesh_mode: str = Field("relief_plane", validation_alias="DEPTH_MESH_MODE")
    depth_grid_resolution: int = Field(256, validation_alias="DEPTH_GRID_RESOLUTION")
    depth_edge_cleanup_enabled: bool = Field(True, validation_alias="DEPTH_EDGE_CLEANUP_ENABLED")
    depth_preview_only: bool = Field(True, validation_alias="DEPTH_PREVIEW_ONLY")
    depth_studio_require_explicit_final_override: bool = Field(True, validation_alias="DEPTH_STUDIO_REQUIRE_EXPLICIT_FINAL_OVERRIDE")

    # --- PHASE 6.1: SAM2 SEGMENTATION ---
    segmentation_method: str = Field("legacy", validation_alias="SEGMENTATION_METHOD")
    sam2_enabled: bool = Field(False, validation_alias="SAM2_ENABLED")
    sam2_device: str = Field("cuda", validation_alias="SAM2_DEVICE")
    sam2_model_cfg: str = Field("sam2_hiera_l.yaml", validation_alias="SAM2_MODEL_CFG")
    sam2_checkpoint: str = Field("models/sam2/sam2_hiera_large.pt", validation_alias="SAM2_CHECKPOINT")
    sam2_fallback_to_legacy: bool = Field(True, validation_alias="SAM2_FALLBACK_TO_LEGACY")
    sam2_review_only: bool = Field(True, validation_alias="SAM2_REVIEW_ONLY")
    sam2_prompt_mode: str = Field("center_box", validation_alias="SAM2_PROMPT_MODE")
    sam2_mode: str = Field("image", validation_alias="SAM2_MODE")
    sam2_max_frames: int = Field(0, validation_alias="SAM2_MAX_FRAMES")

    # --- PHASE A: AR MASK PREVIEW ---
    sam_mask_preview_enabled: bool = Field(False, validation_alias="SAM_MASK_PREVIEW_ENABLED")
    segmentation_preview_provider: str = Field("legacy", validation_alias="SEGMENTATION_PREVIEW_PROVIDER")
    sam_mask_min_confidence: float = Field(0.75, validation_alias="SAM_MASK_MIN_CONFIDENCE")
    sam_mask_preview_timeout_sec: int = Field(5, validation_alias="SAM_MASK_PREVIEW_TIMEOUT_SEC")
    sam_mask_preview_max_image_size: int = Field(768, validation_alias="SAM_MASK_PREVIEW_MAX_IMAGE_SIZE")
    sam_mask_preview_review_only: bool = Field(True, validation_alias="SAM_MASK_PREVIEW_REVIEW_ONLY")

    # --- SAM3 SEGMENTATION ---
    sam3_enabled: bool = Field(False, validation_alias="SAM3_ENABLED")
    sam3_device: str = Field("cuda", validation_alias="SAM3_DEVICE")
    sam3_mode: str = Field("video", validation_alias="SAM3_MODE")
    sam3_text_prompt: str = Field("product", validation_alias="SAM3_TEXT_PROMPT")
    sam3_checkpoint: str = Field("", validation_alias="SAM3_CHECKPOINT")
    sam3_config: str = Field("", validation_alias="SAM3_CONFIG")
    sam3_fallback_to_sam2: bool = Field(True, validation_alias="SAM3_FALLBACK_TO_SAM2")
    sam3_review_only: bool = Field(True, validation_alias="SAM3_REVIEW_ONLY")
    sam3_max_keyframes: int = Field(3, validation_alias="SAM3_MAX_KEYFRAMES")

    # --- AI 3D GENERATION (Phase B/C Scaffolds) ---
    ai_3d_provider: str = Field("none", validation_alias="AI_3D_PROVIDER")
    ai_3d_preview_enabled: bool = Field(False, validation_alias="AI_3D_PREVIEW_ENABLED")

    # --- AI 3D GENERATION — SF3D integration (all default false) ---
    ai_3d_generation_enabled: bool = Field(False, validation_alias="AI_3D_GENERATION_ENABLED")
    ai_3d_default_provider: str = Field("sf3d", validation_alias="AI_3D_DEFAULT_PROVIDER")
    ai_3d_output_format: str = Field("glb", validation_alias="AI_3D_OUTPUT_FORMAT")
    ai_3d_require_review: bool = Field(True, validation_alias="AI_3D_REQUIRE_REVIEW")
    ai_3d_preprocess_enabled: bool = Field(True, validation_alias="AI_3D_PREPROCESS_ENABLED")
    ai_3d_postprocess_enabled: bool = Field(True, validation_alias="AI_3D_POSTPROCESS_ENABLED")

    sf3d_enabled: bool = Field(False, validation_alias="SF3D_ENABLED")
    sf3d_python_path: str = Field(
        "external/stable-fast-3d/.venv_sf3d/Scripts/python.exe",
        validation_alias="SF3D_PYTHON_PATH",
    )
    sf3d_worker_script: str = Field(
        "scripts/sf3d_worker.py",
        validation_alias="SF3D_WORKER_SCRIPT",
    )
    sf3d_device: str = Field("auto", validation_alias="SF3D_DEVICE")
    sf3d_timeout_sec: int = Field(300, validation_alias="SF3D_TIMEOUT_SEC")
    sf3d_input_size: int = Field(512, validation_alias="SF3D_INPUT_SIZE")
    sf3d_texture_resolution: int = Field(1024, validation_alias="SF3D_TEXTURE_RESOLUTION")
    sf3d_remesh: str = Field("none", validation_alias="SF3D_REMESH")
    sf3d_output_format: str = Field("glb", validation_alias="SF3D_OUTPUT_FORMAT")
    sf3d_require_review: bool = Field(True, validation_alias="SF3D_REQUIRE_REVIEW")

    # --- SF3D execution mode (Phase 4D) ---
    # disabled | local_windows | wsl_subprocess | remote_http  (default: disabled)
    sf3d_execution_mode: str = Field("disabled", validation_alias="SF3D_EXECUTION_MODE")
    sf3d_wsl_distro: str = Field("Ubuntu-24.04", validation_alias="SF3D_WSL_DISTRO")
    sf3d_wsl_python_path: str = Field(
        "/home/lenovo/sf3d_venv/bin/python",
        validation_alias="SF3D_WSL_PYTHON_PATH",
    )
    sf3d_wsl_repo_root: str = Field(
        "/mnt/c/Users/Lenovo/.gemini/antigravity/scratch/Qar_imageProccesing",
        validation_alias="SF3D_WSL_REPO_ROOT",
    )
    sf3d_wsl_timeout_sec: int = Field(600, validation_alias="SF3D_WSL_TIMEOUT_SEC")
    sf3d_wsl_output_copy_enabled: bool = Field(True, validation_alias="SF3D_WSL_OUTPUT_COPY_ENABLED")

    sam3d_enabled: bool = Field(False, validation_alias="SAM3D_ENABLED")
    sam3d_output_format: str = Field("glb", validation_alias="SAM3D_OUTPUT_FORMAT")
    sam3d_require_review: bool = Field(True, validation_alias="SAM3D_REQUIRE_REVIEW")
    sam3d_device: str = Field("cuda", validation_alias="SAM3D_DEVICE")
    sam3d_checkpoint: str = Field("", validation_alias="SAM3D_CHECKPOINT")
    sam3d_config: str = Field("", validation_alias="SAM3D_CONFIG")

    meshy_enabled: bool = Field(False, validation_alias="MESHY_ENABLED")
    meshy_api_key: str = Field("", validation_alias="MESHY_API_KEY")
    meshy_require_review: bool = Field(True, validation_alias="MESHY_REQUIRE_REVIEW")

    # --- PHASE 6.2: DEPTH ANYTHING (scaffold only, disabled) ---
    depth_anything_enabled: bool = Field(False, validation_alias="DEPTH_ANYTHING_ENABLED")
    depth_anything_device: str = Field("cuda", validation_alias="DEPTH_ANYTHING_DEVICE")
    depth_anything_model: str = Field("depth-anything-v2-small", validation_alias="DEPTH_ANYTHING_MODEL")
    depth_anything_checkpoint: str = Field(
        "models/depth_anything/depth_anything_v2_small.pth",
        validation_alias="DEPTH_ANYTHING_CHECKPOINT",
    )
    depth_anything_fallback_to_none: bool = Field(True, validation_alias="DEPTH_ANYTHING_FALLBACK_TO_NONE")
    depth_anything_review_only: bool = Field(True, validation_alias="DEPTH_ANYTHING_REVIEW_ONLY")
    depth_anything_max_frames: int = Field(0, validation_alias="DEPTH_ANYTHING_MAX_FRAMES")

    # Depth prior quality gates — depth prior is ONLY allowed when
    # segmentation quality is already high.
    depth_prior_min_segmentation_iou: float = Field(0.85, validation_alias="DEPTH_PRIOR_MIN_SEGMENTATION_IOU")
    depth_prior_max_leakage_ratio: float = Field(0.05, validation_alias="DEPTH_PRIOR_MAX_LEAKAGE_RATIO")
    depth_prior_min_mask_confidence: float = Field(0.75, validation_alias="DEPTH_PRIOR_MIN_MASK_CONFIDENCE")

    # --- AI COMPLETION POLICY (not implemented, thresholds only) ---
    ai_completion_enabled: bool = Field(False, validation_alias="AI_COMPLETION_ENABLED")
    min_observed_surface_for_completion: float = Field(0.50, validation_alias="MIN_OBSERVED_SURFACE_FOR_COMPLETION")
    min_observed_surface_for_production: float = Field(0.70, validation_alias="MIN_OBSERVED_SURFACE_FOR_PRODUCTION")
    max_synthesized_surface_for_review: float = Field(0.50, validation_alias="MAX_SYNTHESIZED_SURFACE_FOR_REVIEW")
    max_synthesized_surface_for_production: float = Field(0.20, validation_alias="MAX_SYNTHESIZED_SURFACE_FOR_PRODUCTION")
    critical_region_completion_allowed: bool = Field(False, validation_alias="CRITICAL_REGION_COMPLETION_ALLOWED")
    generative_completion_default_status: str = Field("review_ready", validation_alias="GENERATIVE_COMPLETION_DEFAULT_STATUS")

    @property
    def is_dev(self) -> bool:
        return self.env == AppEnvironment.LOCAL_DEV

    def check_ml_deps(self) -> list[str]:
        """Returns list of missing ML dependencies."""
        import importlib.util
        missing = []
        for dep in ["rembg", "onnxruntime"]:
            if importlib.util.find_spec(dep) is None:
                missing.append(dep)
        return missing

    def check_processing_deps(self) -> list[str]:
        """Returns list of missing critical processing dependencies."""
        import importlib.util
        missing = []
        for dep in ["fast_simplification"]:
            if importlib.util.find_spec(dep) is None:
                missing.append(dep)
        return missing

    def probe_colmap_binary(self) -> dict:
        """
        Probe COLMAP binary with --help to verify it is actually executable,
        not merely present on disk.

        Returns:
            dict with keys:
                ok           (bool)  — True if the binary ran without OS error
                version_line (str|None) — first line of output, useful for logs
                error        (str|None) — human-readable failure reason
        """
        import subprocess
        binary = Path(self.colmap_path)
        if not binary.exists():
            return {
                "ok": False,
                "version_line": None,
                "error": f"Binary not found at {self.colmap_path}",
            }
        try:
            result = subprocess.run(
                [str(binary), "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # COLMAP may exit 0 or 1 after --help; both mean the binary ran.
            combined = (result.stdout + result.stderr).strip()
            first_line = combined.splitlines()[0] if combined else "(no output)"
            return {"ok": True, "version_line": first_line, "error": None}
        except FileNotFoundError as exc:
            return {"ok": False, "version_line": None, "error": f"OS cannot find binary: {exc}"}
        except PermissionError as exc:
            return {"ok": False, "version_line": None, "error": f"Permission denied: {exc}"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "version_line": None, "error": "COLMAP --help timed out (>10s)"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "version_line": None, "error": str(exc)}

    def probe_openmvs_binaries(self) -> dict:
        """
        Probe OpenMVS binaries to verify they are executable.
        Checks for InterfaceCOLMAP, TextureMesh, DensifyPointCloud, and ReconstructMesh.
        """
        import subprocess
        bin_dir = Path(self.openmvs_path)
        
        if not bin_dir.exists() or not bin_dir.is_dir():
            return {
                "ok": False,
                "error": f"OpenMVS binary directory not found at {self.openmvs_path}",
                "missing_binaries": ["InterfaceCOLMAP", "TextureMesh", "DensifyPointCloud", "ReconstructMesh"]
            }

        binaries_to_check = ["InterfaceCOLMAP", "TextureMesh", "DensifyPointCloud", "ReconstructMesh"]
        missing = []
        
        # On non-Windows, check without .exe
        is_windows = __import__("os").name == "nt"
        
        for bin_name in binaries_to_check:
            exe_name = bin_name + ".exe" if is_windows else bin_name
            bin_path = bin_dir / exe_name
            
            if not bin_path.exists():
                # Fallback check without extension on Windows just in case
                if is_windows and (bin_dir / bin_name).exists():
                    bin_path = bin_dir / bin_name
                else:
                    missing.append(bin_name)
                    continue

            try:
                # Running with --help to see if it executes
                subprocess.run([str(bin_path), "--help"], capture_output=True, text=True, timeout=5)
            except Exception:
                missing.append(bin_name)

        if missing:
            return {
                "ok": False,
                "error": f"Missing or unexecutable OpenMVS binaries: {', '.join(missing)}",
                "missing_binaries": missing
            }

        return {"ok": True, "error": None, "missing_binaries": []}

    def check_free_disk_gb(self) -> float:
        """
        Returns the free disk space (in GB) on the partition that hosts data_root.
        Returns float('inf') if the query fails (permissive fallback).
        """
        import shutil
        try:
            path = Path(self.data_root).resolve()
            # Create the directory if it doesn't exist so shutil.disk_usage works.
            path.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(path)
            return usage.free / (1024 ** 3)
        except Exception:  # noqa: BLE001
            return float("inf")

    def probe_ffmpeg(self) -> dict:
        """Probe FFmpeg binary with -version."""
        return self._probe_binary(self.ffmpeg_path, ["-version"])

    def probe_ffprobe(self) -> dict:
        """Probe ffprobe binary with -version."""
        return self._probe_binary(self.ffprobe_path, ["-version"])

    def _probe_binary(self, path: Optional[str], args: list[str]) -> dict:
        """Generic binary prober."""
        import subprocess
        if not path:
            return {"ok": False, "version_line": None, "error": "Path not configured"}
        
        resolved_path = self.resolve_executable(path)
        if not resolved_path:
            return {"ok": False, "version_line": None, "error": f"Binary not found or not in PATH: {path}"}
            
        try:
            result = subprocess.run(
                [str(resolved_path)] + args,
                capture_output=True,
                text=True,
                timeout=5,
            )
            combined = (result.stdout + result.stderr).strip()
            first_line = combined.splitlines()[0] if combined else "(no output)"
            return {"ok": True, "version_line": first_line, "error": None}
        except Exception as exc:
            return {"ok": False, "version_line": None, "error": str(exc)}

    def resolve_executable(self, path_or_command: str) -> Optional[str]:
        """
        Resolves a binary path. 
        1. Checks if it exists as an absolute/relative path.
        2. Checks if it is available in the system PATH via shutil.which.
        
        Returns the resolved absolute path as a string, or None if not found.
        """
        import shutil
        if not path_or_command:
            return None
            
        # 1. Direct path check
        p = Path(path_or_command)
        if p.exists():
            return str(p.resolve())
            
        # 2. PATH resolution
        resolved = shutil.which(path_or_command)
        if resolved:
            return str(Path(resolved).resolve())
            
        return None

    def validate_setup(self):
        """Validates that the current environment has all necessary configuration."""
        if self.env == AppEnvironment.PILOT or self.env == AppEnvironment.PRODUCTION:
            if not self.pilot_api_key:
                raise ValueError(f"PILOT_API_KEY is mandatory in {getattr(self.env, 'value', self.env)} environment.")


            cp = Path(self.colmap_path)
            if not cp.exists():
                if not cp.with_suffix(".bat").exists() and not cp.with_suffix(".exe").exists():
                    raise ValueError(f"COLMAP binary not found at {self.colmap_path}")

            # Dependency Validation
            if self.strict_ml_segmentation:
                missing_ml = self.check_ml_deps()
                if missing_ml:
                    import logging
                    logger = logging.getLogger("settings")
                    logger.warning(
                        f"CRITICAL: Missing ML dependencies in {getattr(self.env, 'value', self.env)}: {missing_ml}"
                    )
            
            # Pilot/Production Guard for Simulated Pipeline
            if self.recon_pipeline == ReconstructionPipeline.SIMULATED.value:
                raise ValueError(f"Simulated reconstruction pipeline is strictly prohibited in {getattr(self.env, 'value', self.env)} environment.")


# Singleton instance
settings = Settings()
