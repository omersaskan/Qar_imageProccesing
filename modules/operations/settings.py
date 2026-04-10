from enum import Enum
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

class AppEnvironment(str, Enum):
    LOCAL_DEV = "local_dev"
    PILOT = "pilot"
    PRODUCTION = "production"

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

    @property
    def is_dev(self) -> bool:
        return self.env == AppEnvironment.LOCAL_DEV

    def validate_setup(self):
        """Validates that the current environment has all necessary configuration."""
        if self.env in [AppEnvironment.PILOT, AppEnvironment.PRODUCTION]:
            if not self.pilot_api_key:
                raise ValueError(f"PILOT_API_KEY is mandatory in {self.env.value} environment.")
            
            cp = Path(self.colmap_path)
            if not cp.exists():
                if not cp.with_suffix(".bat").exists() and not cp.with_suffix(".exe").exists():
                    raise ValueError(f"COLMAP binary not found at {self.colmap_path}")

# Singleton instance
settings = Settings()
