"""
Quality gate for AI 3D generation results.

Evaluates provider result + output existence and assigns a quality verdict:
  ok          — output exists, provider succeeded, no forced review
  review      — output exists but review required by policy
  unavailable — provider was unavailable
  failed      — provider failed or output missing

Always includes the provenance warnings:
  ai_generated_not_true_scan
  generated_geometry_estimated
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional


_PROVENANCE_WARNINGS = [
    "ai_generated_not_true_scan",
    "generated_geometry_estimated",
]


def evaluate(
    provider_result: Dict[str, Any],
    output_glb_path: Optional[str],
    review_required: bool = True,
) -> Dict[str, Any]:
    """
    Returns a quality gate dict:
      verdict        : "ok" | "review" | "unavailable" | "failed"
      output_exists  : bool
      warnings       : list[str]
      reason         : str | None
    """
    warnings = list(_PROVENANCE_WARNINGS)
    status = provider_result.get("status", "failed")

    output_exists = bool(output_glb_path and Path(output_glb_path).exists())

    if status in ("unavailable", "busy"):
        return _gate("unavailable", output_exists, warnings,
                     reason=(
                         provider_result.get("error_code")
                         or provider_result.get("error")
                         or "provider_unavailable"
                     ))

    if status != "ok":
        return _gate("failed", output_exists, warnings,
                     reason=provider_result.get("error") or "provider_failed")

    if not output_exists:
        return _gate("failed", False, warnings, reason="output_glb_missing")

    if review_required:
        warnings.append("review_required")
        return _gate("review", True, warnings, reason="ai_generated_asset_requires_review")

    return _gate("ok", True, warnings, reason=None)


def _gate(verdict: str, output_exists: bool,
          warnings: list, reason: Optional[str]) -> Dict[str, Any]:
    return {
        "verdict": verdict,
        "output_exists": output_exists,
        "warnings": warnings,
        "reason": reason,
    }
