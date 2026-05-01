"""
AI completion — generative 3D for unobserved surfaces.

Photogrammetry stays primary.  When the captured surface coverage is low
(e.g. forklift bottom, asansör arkası, çiçek pot içi) and the policy
allows it, this module hands a *guidance* asset (the reconstructed mesh
+ a representative image) to a generative provider (Hunyuan3D-2,
SAM3D, Meshy) and merges the result back, while strict gates clamp the
synthesized surface ratio to per-profile limits.
"""

from .base import (
    CompletionRequest,
    CompletionResult,
    CompletionProvider,
    CompletionStatus,
)
from .coverage import compute_observed_surface_ratio
from .policy import (
    CompletionDecision,
    decide_completion_path,
    apply_quality_gates,
)
from .service import AICompletionService, build_default_service

__all__ = [
    "CompletionRequest",
    "CompletionResult",
    "CompletionProvider",
    "CompletionStatus",
    "compute_observed_surface_ratio",
    "CompletionDecision",
    "decide_completion_path",
    "apply_quality_gates",
    "AICompletionService",
    "build_default_service",
]
