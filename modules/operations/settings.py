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
    openmvs_path: str = Field(r"C:\openmvs\bin", validation_alias="OPENMVS_BIN_PATH")
    use_gpu: bool = Field(True, validation_alias="RECON_USE_GPU")

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
    
    # Default effort will use whatever is on disk (usually from extraction rate=15).
    # Fallback DENSER_FRAMES will re-extract with this rate.
    recon_fallback_sample_rate: int = Field(5, validation_alias="RECON_FALLBACK_SAMPLE_RATE")

    # For denser_frames: 1.0 means use all frames, lower means more sparse. 
    recon_denser_sampling_ratio: float = Field(1.0, validation_alias="RECON_DENSER_SAMPLING_RATIO")

    # --- SPRINT 1: TICKET-004 — Disk Space Preflight ---
    # Minimum free disk space on the data_root partition before an upload is
    # accepted. Stricter environments should raise this. Default: 5 GB.
    min_free_disk_gb: float = Field(5.0, validation_alias="MIN_FREE_DISK_GB")

    # --- RECONSTRUCTION ENGINES ---
    # Global switch for the default pipeline: "colmap_dense" or "colmap_openmvs"
    recon_pipeline: str = Field("colmap_dense", validation_alias="RECON_PIPELINE")
    
    # Poisson mesher timeout in seconds.
    recon_poisson_timeout_sec: int = Field(300, validation_alias="RECON_POISSON_TIMEOUT_SEC")
    
    # OpenMVS specific flags
    openmvs_fail_hard: bool = Field(False, validation_alias="OPENMVS_FAIL_HARD")
    openmvs_textured_output: bool = Field(True, validation_alias="OPENMVS_TEXTURED_OUTPUT")
    require_textured_output: bool = Field(False, validation_alias="REQUIRE_TEXTURED_OUTPUT")

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


# Singleton instance
settings = Settings()
