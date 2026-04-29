from enum import Enum
from typing import Optional
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

    # Binaries
    colmap_path: str = Field(r"C:\colmap\colmap.exe", validation_alias="RECON_ENGINE_PATH")
    openmvs_path: str = Field(
        r"C:\openmvs\bin", 
        validation_alias="OPENMVS_BIN_PATH"
    )
    ffmpeg_path: str = Field(
        r"C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe", 
        validation_alias="FFMPEG_PATH"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # SPRINT: OpenMVS env harmonization
        # OPENMVS_BIN_PATH wins, but we fallback to OPENMVS_BIN if set
        import os
        if not os.environ.get("OPENMVS_BIN_PATH") and os.environ.get("OPENMVS_BIN"):
            self.openmvs_path = os.environ.get("OPENMVS_BIN")
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
    min_video_duration_sec: float = Field(2.0, validation_alias="MIN_VIDEO_DURATION_SEC")
    max_video_duration_sec: float = Field(120.0, validation_alias="MAX_VIDEO_DURATION_SEC")
    min_video_width: int = Field(720, validation_alias="MIN_VIDEO_WIDTH")
    min_video_height: int = Field(720, validation_alias="MIN_VIDEO_HEIGHT")
    min_video_fps: float = Field(20.0, validation_alias="MIN_VIDEO_FPS")
    
    # --- AR CAPTURE QUALITY GATING ---
    ar_min_coverage: float = Field(90.0, validation_alias="AR_MIN_COVERAGE")
    ar_max_gap: float = Field(45.0, validation_alias="AR_MAX_GAP")
    ar_min_accepted_frames: int = Field(100, validation_alias="AR_MIN_ACCEPTED_FRAMES")
    ar_max_blur_ratio: float = Field(0.3, validation_alias="AR_MAX_BLUR_RATIO")
    ar_min_duration_sec: float = Field(15.0, validation_alias="AR_MIN_DURATION_SEC")

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

    # --- TEXTURE QUALITY QA ---
    max_black_pixel_ratio: float = Field(0.20, validation_alias="MAX_BLACK_PIXEL_RATIO")
    max_dominant_background_ratio: float = Field(0.15, validation_alias="MAX_DOMINANT_BACKGROUND_RATIO")
    min_atlas_coverage_ratio: float = Field(0.60, validation_alias="MIN_ATLAS_COVERAGE_RATIO")
    min_near_white_ratio_white_cream: float = Field(0.40, validation_alias="MIN_NEAR_WHITE_RATIO_WHITE_CREAM")
    white_cream_max_background_ratio: Optional[float] = Field(None, validation_alias="WHITE_CREAM_MAX_BACKGROUND_RATIO")
    expected_product_color: str = Field("unknown", validation_alias="EXPECTED_PRODUCT_COLOR")
    
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
