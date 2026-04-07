from enum import Enum
from pydantic import BaseModel

class CleanupProfileType(str, Enum):
    MOBILE_LOW = "mobile_low"
    MOBILE_DEFAULT = "mobile_default"
    HQ_REFERENCE = "hq_reference"

class CleanupProfile(BaseModel):
    name: CleanupProfileType
    target_polycount: int
    decimation_ratio: float
    texture_size: int
    remove_noise: bool = True

PROFILES = {
    CleanupProfileType.MOBILE_LOW: CleanupProfile(
        name=CleanupProfileType.MOBILE_LOW,
        target_polycount=20000,
        decimation_ratio=0.1,
        texture_size=1024
    ),
    CleanupProfileType.MOBILE_DEFAULT: CleanupProfile(
        name=CleanupProfileType.MOBILE_DEFAULT,
        target_polycount=50000,
        decimation_ratio=0.3,
        texture_size=2048
    ),
    CleanupProfileType.HQ_REFERENCE: CleanupProfile(
        name=CleanupProfileType.HQ_REFERENCE,
        target_polycount=200000,
        decimation_ratio=1.0,
        texture_size=4096,
        remove_noise=False
    )
}
