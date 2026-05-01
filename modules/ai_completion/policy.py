"""Quality gates that decide if/when AI completion is allowed and how its result grades."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base import CompletionStatus


@dataclass
class CompletionDecision:
    should_run: bool
    reason: str
    observed_surface_ratio: float
    min_observed_for_completion: float
    min_observed_for_production: float
    max_synth_for_review: float
    max_synth_for_production: float
    target_status: CompletionStatus = CompletionStatus.SKIPPED_DISABLED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_run": self.should_run,
            "reason": self.reason,
            "observed_surface_ratio": self.observed_surface_ratio,
            "min_observed_for_completion": self.min_observed_for_completion,
            "min_observed_for_production": self.min_observed_for_production,
            "max_synth_for_review": self.max_synth_for_review,
            "max_synth_for_production": self.max_synth_for_production,
            "target_status": self.target_status.value,
        }


def decide_completion_path(
    observed_ratio: float,
    settings_obj: Any,
    capture_profile: Optional[Any] = None,
    provider_name: str = "none",
) -> CompletionDecision:
    """
    Decide whether to invoke the generative provider, and what target status
    a successful completion would earn.

    Decision tree:
        1. ai_completion_enabled=false           → SKIPPED_DISABLED
        2. provider == 'none'                    → SKIPPED_PROVIDER_NONE
        3. observed >= production threshold      → SKIPPED_SUFFICIENT (no need)
        4. observed >= completion threshold      → run; target REVIEW_READY
        5. observed >= 0.30                      → run; target PREVIEW_ONLY
        6. else                                  → don't run; FAILED (capture unsalvageable)

    `capture_profile` (when supplied) overrides the production / review
    thresholds — large_mounted accepts %55 prod surface vs default %70.
    """
    min_completion = float(getattr(settings_obj, "min_observed_surface_for_completion", 0.50))
    min_production = float(
        getattr(capture_profile, "min_observed_surface_for_production", None)
        or getattr(settings_obj, "min_observed_surface_for_production", 0.70)
    )
    min_review = float(
        getattr(capture_profile, "min_observed_surface_for_review", None)
        or 0.50
    )
    max_synth_review = float(getattr(settings_obj, "max_synthesized_surface_for_review", 0.50))
    max_synth_production = float(getattr(settings_obj, "max_synthesized_surface_for_production", 0.20))
    enabled = bool(getattr(settings_obj, "ai_completion_enabled", False))

    base_kwargs = dict(
        observed_surface_ratio=observed_ratio,
        min_observed_for_completion=min_completion,
        min_observed_for_production=min_production,
        max_synth_for_review=max_synth_review,
        max_synth_for_production=max_synth_production,
    )

    if not enabled:
        return CompletionDecision(
            should_run=False,
            reason="ai_completion_enabled=false",
            target_status=CompletionStatus.SKIPPED_DISABLED,
            **base_kwargs,
        )

    if provider_name == "none":
        return CompletionDecision(
            should_run=False,
            reason="AI_3D_PROVIDER=none",
            target_status=CompletionStatus.SKIPPED_PROVIDER_NONE,
            **base_kwargs,
        )

    if observed_ratio >= min_production:
        return CompletionDecision(
            should_run=False,
            reason=f"observed={observed_ratio:.2f} ≥ production_threshold={min_production:.2f}",
            target_status=CompletionStatus.SKIPPED_SUFFICIENT,
            **base_kwargs,
        )

    if observed_ratio >= min_completion:
        return CompletionDecision(
            should_run=True,
            reason=f"observed={observed_ratio:.2f} in [{min_completion:.2f}, {min_production:.2f})",
            target_status=CompletionStatus.REVIEW_READY,
            **base_kwargs,
        )

    if observed_ratio >= 0.30:
        return CompletionDecision(
            should_run=True,
            reason=f"observed={observed_ratio:.2f} too low for review; preview-only",
            target_status=CompletionStatus.PREVIEW_ONLY,
            **base_kwargs,
        )

    return CompletionDecision(
        should_run=False,
        reason=f"observed={observed_ratio:.2f} < 0.30 — capture unsalvageable; recapture needed",
        target_status=CompletionStatus.FAILED,
        **base_kwargs,
    )


def apply_quality_gates(
    decision: CompletionDecision,
    synthesized_ratio: float,
) -> CompletionStatus:
    """
    After the provider returns, downgrade the target status if the
    synthesized fraction exceeds the per-tier ceiling.
    """
    if decision.target_status == CompletionStatus.PRODUCTION_CANDIDATE:
        if synthesized_ratio > decision.max_synth_for_production:
            return CompletionStatus.REVIEW_READY \
                if synthesized_ratio <= decision.max_synth_for_review \
                else CompletionStatus.PREVIEW_ONLY
        return CompletionStatus.PRODUCTION_CANDIDATE

    if decision.target_status == CompletionStatus.REVIEW_READY:
        if synthesized_ratio > decision.max_synth_for_review:
            return CompletionStatus.PREVIEW_ONLY
        return CompletionStatus.REVIEW_READY

    return decision.target_status
