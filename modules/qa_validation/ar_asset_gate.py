"""
AR asset publish gate — Sprint 7.

Combines glTF-Transform optimization result + Khronos Validator result
into a publish verdict.

Verdict logic:
  - validator status=error       → reject (do not publish)
  - validator status=warning     → review
  - optimizer status=failed      → review (warn but don't block)
  - optimizer status=unavailable → pass (tool not installed; non-blocking)
  - all ok                       → pass
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ArAssetGateResult:
    verdict: str                   # pass | review | reject
    reasons: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    validation_status: Optional[str] = None
    optimization_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_ar_gate(
    optimizer_result,    # GltfTransformResult or None
    validator_result,    # GltfValidationReport or None
) -> ArAssetGateResult:
    """
    Produce a publish verdict from optimizer + validator results.
    """
    reasons: List[str] = []
    suggestions: List[str] = []
    reject = False
    review = False

    opt_status = getattr(optimizer_result, "status", None) if optimizer_result else None
    val_status = getattr(validator_result, "status", None) if validator_result else None

    # Validator gating
    if val_status == "error":
        ec = getattr(validator_result, "error_count", 0)
        reasons.append(f"glTF validation errors: {ec}")
        suggestions.append("Fix glTF validation errors before publishing to AR")
        reject = True
    elif val_status == "warning":
        wc = getattr(validator_result, "warning_count", 0)
        reasons.append(f"glTF validation warnings: {wc}")
        suggestions.append("Review glTF warnings; asset may have compatibility issues on some devices")
        review = True
    elif val_status == "unavailable":
        reasons.append("glTF validator not installed — validation skipped")

    # Optimizer status (non-blocking)
    if opt_status == "failed":
        reasons.append("glTF-Transform optimization failed")
        suggestions.append("Check gltf-transform installation or input GLB integrity")
        review = True
    elif opt_status == "unavailable":
        reasons.append("glTF-Transform not installed — optimization skipped")

    if reject:
        verdict = "reject"
    elif review:
        verdict = "review"
    else:
        verdict = "pass"

    return ArAssetGateResult(
        verdict=verdict,
        reasons=reasons,
        suggestions=suggestions,
        validation_status=val_status,
        optimization_status=opt_status,
    )
