"""
Depth Prior Policy — Decision rules for when depth priors are allowed.

⚠️  Depth Anything is NOT a replacement for good segmentation.
Depth prior is ONLY allowed after segmentation quality is confirmed.

Principle: segmentation is the bottleneck. Do not use depth to fix dirty masks.
"""

import logging
from typing import Dict, Any
from modules.operations.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coverage / Completion Policy Constants
# ---------------------------------------------------------------------------

COVERAGE_PRODUCTION = 0.70    # >= 70% observed → production candidate
COVERAGE_REVIEW = 0.50       # 50-70% → review_ready, AI completion allowed
COVERAGE_PREVIEW = 0.30      # 30-50% → preview_only / concept_only
# < 30% → fail by default


def evaluate_depth_prior_eligibility(
    segmentation_iou: float,
    leakage_ratio: float,
    mask_confidence: float,
) -> Dict[str, Any]:
    """
    Evaluate whether depth prior is allowed based on segmentation quality.

    Returns:
        dict with:
            depth_prior_allowed: bool
            reason: str
            segmentation_iou: float
            leakage_ratio: float
            mask_confidence: float
            thresholds: dict of threshold values used
    """
    min_iou = settings.depth_prior_min_segmentation_iou
    max_leak = settings.depth_prior_max_leakage_ratio
    min_conf = settings.depth_prior_min_mask_confidence

    reasons = []
    if segmentation_iou < min_iou:
        reasons.append(f"IoU {segmentation_iou:.3f} < {min_iou}")
    if leakage_ratio > max_leak:
        reasons.append(f"leakage {leakage_ratio:.3f} > {max_leak}")
    if mask_confidence < min_conf:
        reasons.append(f"confidence {mask_confidence:.3f} < {min_conf}")

    allowed = len(reasons) == 0 and settings.depth_anything_enabled

    if not settings.depth_anything_enabled:
        reasons.append("DEPTH_ANYTHING_ENABLED=false")

    reason = "Segmentation quality sufficient" if allowed else (
        "Segmentation quality insufficient: " + "; ".join(reasons)
    )

    return {
        "depth_prior_allowed": allowed,
        "reason": reason,
        "segmentation_iou": segmentation_iou,
        "leakage_ratio": leakage_ratio,
        "mask_confidence": mask_confidence,
        "thresholds": {
            "min_segmentation_iou": min_iou,
            "max_leakage_ratio": max_leak,
            "min_mask_confidence": min_conf,
        },
    }


def classify_coverage(observed_surface_ratio: float) -> Dict[str, Any]:
    """
    Classify asset readiness based on observed surface coverage.

    Coverage policy:
    - >= 0.70 → production_candidate
    - 0.50–0.70 → review_ready (AI completion allowed)
    - 0.30–0.50 → preview_only / concept_only
    - < 0.30 → failed (concept_only if user explicitly accepts)

    Critical regions (logo, label, text, brand marks, dimensions)
    must NOT be hallucinated.  Synthesized geometry/texture must
    be marked in metadata.
    """
    if observed_surface_ratio >= COVERAGE_PRODUCTION:
        return {
            "status": "production_candidate",
            "observed_surface_ratio": observed_surface_ratio,
            "ai_completion_allowed": False,
            "reason": "Sufficient observed surface for production",
        }
    elif observed_surface_ratio >= COVERAGE_REVIEW:
        return {
            "status": "review_ready",
            "observed_surface_ratio": observed_surface_ratio,
            "ai_completion_allowed": settings.ai_completion_enabled,
            "reason": "Review-ready, AI completion may be considered",
        }
    elif observed_surface_ratio >= COVERAGE_PREVIEW:
        return {
            "status": "preview_only",
            "observed_surface_ratio": observed_surface_ratio,
            "ai_completion_allowed": settings.ai_completion_enabled,
            "reason": "Limited coverage — preview/concept only",
        }
    else:
        return {
            "status": "failed",
            "observed_surface_ratio": observed_surface_ratio,
            "ai_completion_allowed": False,
            "reason": "Insufficient observed surface (<30%)",
        }
