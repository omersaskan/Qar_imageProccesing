"""
Constants and type documentation for the asset quality pipeline.
All pipeline functions return plain dicts for manifest compatibility.
"""

# Score thresholds
SCORE_PRODUCTION_READY = 85
SCORE_MOBILE_READY = 65
SCORE_NEEDS_REVIEW = 40

# Face/vertex count thresholds
FACE_COUNT_HIGH = 200_000
FACE_COUNT_EXCESSIVE = 500_000
VERTEX_COUNT_HIGH = 200_000

# Bounding box plausibility (world units, typically meters)
BOUNDS_MIN_PLAUSIBLE = 0.001
BOUNDS_MAX_PLAUSIBLE = 1000.0

# Default LOD tier definitions
LOD_TIERS_DEFAULT = [
    {"name": "preview",  "target_faces": 10_000,  "recommended": True},
    {"name": "mobile",   "target_faces": 25_000,  "recommended": True},
    {"name": "desktop",  "target_faces": 75_000,  "recommended": False},
]

# Mobile texture limits
MOBILE_MAX_TEXTURES = 4
TEXTURE_RESOLUTION_WARN_MOBILE = 2048
