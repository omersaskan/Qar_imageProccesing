from enum import Enum
from pydantic import BaseModel
from typing import Optional

class CleanupProfileType(str, Enum):
    MOBILE_PREVIEW = "mobile_preview"
    MOBILE_HIGH = "mobile_high"
    DESKTOP_HIGH = "desktop_high"
    RAW_ARCHIVE = "raw_archive"
    TEXTURE_SAFE_COPY = "texture_safe_copy"
    MOBILE_DEFAULT = "mobile_high"

class CleanupProfile(BaseModel):
    name: CleanupProfileType
    target_polycount: int
    face_count_limit: int
    decimation_ratio: float
    texture_size: int
    remove_noise: bool = True
    is_mobile_ready: bool = False

PROFILES = {
    CleanupProfileType.MOBILE_PREVIEW: CleanupProfile(
        name=CleanupProfileType.MOBILE_PREVIEW,
        target_polycount=45000,
        face_count_limit=50000,
        decimation_ratio=0.1,
        texture_size=1024,
        is_mobile_ready=True
    ),
    CleanupProfileType.MOBILE_HIGH: CleanupProfile(
        name=CleanupProfileType.MOBILE_HIGH,
        target_polycount=140000,
        face_count_limit=150000,
        decimation_ratio=0.5,
        texture_size=2048,
        is_mobile_ready=True
    ),
    CleanupProfileType.DESKTOP_HIGH: CleanupProfile(
        name=CleanupProfileType.DESKTOP_HIGH,
        target_polycount=240000,
        face_count_limit=250000,
        decimation_ratio=1.0,
        texture_size=4096,
        is_mobile_ready=False
    ),
    CleanupProfileType.RAW_ARCHIVE: CleanupProfile(
        name=CleanupProfileType.RAW_ARCHIVE,
        target_polycount=2000000,
        face_count_limit=5000000,
        decimation_ratio=1.0,
        texture_size=4096,
        remove_noise=False,
        is_mobile_ready=False
    ),
    CleanupProfileType.TEXTURE_SAFE_COPY: CleanupProfile(
        name=CleanupProfileType.TEXTURE_SAFE_COPY,
        target_polycount=1000000,
        face_count_limit=2000000,
        decimation_ratio=1.0,
        texture_size=4096,
        remove_noise=False,
        is_mobile_ready=False
    )
}
