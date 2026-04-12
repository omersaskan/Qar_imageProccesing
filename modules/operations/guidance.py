"""
modules/operations/guidance.py

SPRINT 3 — TICKET-010: Guidance tuning and message refinement

Changes vs previous version:
- Replaced flat if/elif status mapping with a pattern-based message registry
  that maps (status, failure pattern) → (code, operator-readable message, severity).
- Removed technically-leaking raw failure strings from operator-facing messages;
  they are still preserved in guidance.validation_summary (full report is kept).
- Reduced noisy duplication: FAILURE_REASON and QUALITY_BAR_NOT_MET were emitted
  simultaneously for every recapture; now a single focused message per pattern.
- _enrich_from_coverage: added specific guidance for each known failure pattern
  (motion too narrow, scale/shape variation, not enough readable frames, fallback
  masking, low confidence, ML unavailable).
- _enrich_from_validation: added guidance for component contamination,
  texture/UV failure, material semantics, validation review state.
- to_markdown: added validation summary table, made severity icons consistent.
- All new message codes are documented at top of file for operator runbook reference.

MESSAGE CODE REGISTRY:
  AWAITING_UPLOAD           — session just created, no video yet
  PROCESSING_RECONSTRUCTION — CAPTURED: reconstruction in progress
  PROCESSING_GENERIC        — any intermediate processing stage
  READY_FOR_REVIEW          — asset validated, human review needed
  READY_FOR_PUBLISH         — asset fully passed, will publish
  REVIEW_STUB_ASSET         — stub reconstruction, needs real scan
  SYSTEM_FAILURE_CONFIG     — non-retryable config / env failure
  SYSTEM_FAILURE_PIPELINE   — unexpected pipeline failure
  RECAPTURE_NEEDED          — generic recapture header message
  RECAPTURE_LOW_DIVERSITY   — too few unique horizontal viewpoints
  RECAPTURE_NARROW_MOTION   — object barely moves across frame centers
  RECAPTURE_SCALE_FLAT      — no depth/scale variation across shots
  RECAPTURE_TOO_FEW_FRAMES  — not enough usable frames
  RECAPTURE_MASKING_FALLBACK — too many frames processed with heuristic mask
  RECAPTURE_LOW_CONFIDENCE  — ML confidence too low on too many frames
  MASKING_DEGRADED_ML       — ML segmentation unavailable, results unreliable
  MISSING_TOP_VIEWS         — no elevated-angle shots detected
  CONTAMINATION_HIGH        — 3D model heavily polluted with background
  CONTAMINATION_FRAGMENTED  — model split into multiple disconnected components
  TEXTURE_UV_FAILURE        — UV/tex mapping failed or degraded
  MATERIAL_INFO             — informational: diffuse-only photoscan detected
  VALIDATION_REVIEW_ADVICE  — decision is 'review', human must judge
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from modules.shared_contracts.models import (
    CaptureGuidance, GuidanceSeverity, AssetStatus
)


# ─────────────────────────────────────────────────────────────────────────────
# Severity helpers
# ─────────────────────────────────────────────────────────────────────────────

_INFO = GuidanceSeverity.INFO
_WARN = GuidanceSeverity.WARNING
_CRIT = GuidanceSeverity.CRITICAL


def _msg(code: str, message: str, severity: GuidanceSeverity) -> Dict[str, Any]:
    return {"code": code, "message": message, "severity": severity}


# ─────────────────────────────────────────────────────────────────────────────
# Known failure-reason patterns → operator message
# (matched by substring; order matters — most specific first)
# ─────────────────────────────────────────────────────────────────────────────

_FAILURE_PATTERNS = [
    # Config / env
    ("not configured",      "SYSTEM_FAILURE_CONFIG",
     "The reconstruction engine is not configured. Contact engineering.",        _CRIT),
    ("VIOLATION",           "SYSTEM_FAILURE_CONFIG",
     "A pipeline security constraint was violated. Contact engineering.",        _CRIT),
    ("CUDA",                "SYSTEM_FAILURE_CONFIG",
     "GPU acceleration failed. Check CUDA/driver setup or disable GPU mode.",    _CRIT),
    ("binary not found",    "SYSTEM_FAILURE_CONFIG",
     "COLMAP binary is missing. Check RECON_ENGINE_PATH in settings.",           _CRIT),
    ("Permission denied",   "SYSTEM_FAILURE_CONFIG",
     "Permission error accessing pipeline binary. Contact engineering.",         _CRIT),

    # Recapture - coverage
    ("viewpoint diversity", "RECAPTURE_LOW_DIVERSITY",
     "Not enough unique viewpoints captured. Walk around the object more slowly "
     "and from more distinct angles before pressing record.",                    _CRIT),
    ("object motion",       "RECAPTURE_NARROW_MOTION",
     "The subject barely moved across the frame between shots. Keep the camera "
     "moving continuously and orbit the product in a wider arc.",               _WARN),
    ("scale/shape",         "RECAPTURE_SCALE_FLAT",
     "Very little depth or shape variation detected. Capture more views from "
     "elevated angles (30–60° above the product) and from closer/further.",     _WARN),
    ("readable",            "RECAPTURE_TOO_FEW_FRAMES",
     "Too few usable frames were extracted. Ensure good lighting, slow camera "
     "movement, and a clean background.",                                        _WARN),
    ("heuristic fallback",  "RECAPTURE_MASKING_FALLBACK",
     "Object masks relied on heuristic fallback for most frames. Ensure the "
     "product is clearly separated from the background.",                        _WARN),
    ("semantic confidence", "RECAPTURE_LOW_CONFIDENCE",
     "Object detection confidence was too low on many frames. Use a plain solid "
     "background with strong, even lighting and high contrast.",                 _WARN),

    # Reconstruction
    ("Insufficient masked input", "RECAPTURE_MASKING_FALLBACK",
     "Reconstruction had too few clean masked frames. Re-capture with better "
     "background separation.",                                                   _CRIT),
    ("max retry",           "SYSTEM_FAILURE_PIPELINE",
     "The pipeline failed repeatedly for this session. Contact engineering "
     "with the session ID.",                                                     _CRIT),
    ("timed out",           "SYSTEM_FAILURE_PIPELINE",
     "Processing exceeded the allowed time limit. Check server resources "
     "or contact engineering.",                                                  _CRIT),
]


def _match_failure_reason(reason: str) -> Optional[Dict[str, Any]]:
    """Return the first matching operator message for a raw failure reason string."""
    if not reason:
        return None
    reason_lower = reason.lower()
    for pattern, code, message, severity in _FAILURE_PATTERNS:
        if pattern.lower() in reason_lower:
            return _msg(code, message, severity)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main aggregator
# ─────────────────────────────────────────────────────────────────────────────

class GuidanceAggregator:
    """
    Assembles operator-facing guidance from backend reports and session status.

    Design principles (Sprint 3):
      1. Operator messages never leak raw technical stack traces; full context
         remains in validation_summary / coverage_summary for engineering use.
      2. Each distinct root cause produces exactly ONE message code.
      3. Severity is consistent: CRITICAL = action required, WARNING = take note,
         INFO = status only.
      4. next_action is a clear verb phrase, not a multi-step paragraph.
    """

    def generate_guidance(
        self,
        session_id: str,
        status: AssetStatus,
        coverage_report: Optional[Dict[str, Any]] = None,
        validation_report: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None,
    ) -> CaptureGuidance:

        messages: List[Dict[str, Any]] = []
        should_recapture = False
        is_ready_for_review = False
        next_action = "Processing — no action required."

        # ── Step 1: Base status message ───────────────────────────────────────
        if status == AssetStatus.CREATED:
            next_action = "Upload the product video to begin processing."
            messages.append(_msg(
                "AWAITING_UPLOAD",
                "Waiting for video upload. No action needed until the upload completes.",
                _INFO,
            ))

        elif status == AssetStatus.CAPTURED:
            next_action = "Reconstruction in progress — no action required."
            messages.append(_msg(
                "PROCESSING_RECONSTRUCTION",
                "Frames extracted successfully. 3D reconstruction is running.",
                _INFO,
            ))

        elif status in {
            AssetStatus.RECONSTRUCTED,
            AssetStatus.CLEANED,
            AssetStatus.EXPORTED,
        }:
            next_action = "Asset processing in progress — no action required."
            messages.append(_msg(
                "PROCESSING_GENERIC",
                f"Pipeline stage: {status.value}. System is working automatically.",
                _INFO,
            ))

        elif status == AssetStatus.RECAPTURE_REQUIRED:
            next_action = "Re-capture required — read the instructions below before retaking."
            should_recapture = True
            # Header message — only one, not duplicated by failure_reason below
            messages.append(_msg(
                "RECAPTURE_NEEDED",
                "This capture did not meet production quality standards. "
                "Review the instructions below and retake the video.",
                _CRIT,
            ))
            # Pattern-matched detail
            detail = _match_failure_reason(failure_reason or "")
            if detail and detail["code"] != "RECAPTURE_NEEDED":
                messages.append(detail)

        elif status == AssetStatus.VALIDATED:
            final_decision = (validation_report or {}).get("final_decision", "unknown")
            if final_decision == "pass":
                next_action = "Asset passed validation — publishing automatically."
                is_ready_for_review = True
                messages.append(_msg(
                    "READY_FOR_PUBLISH",
                    "Asset passed all quality checks and will be published.",
                    _INFO,
                ))
            elif final_decision == "review":
                next_action = "Open the dashboard and visually inspect the 3D model before approving."
                is_ready_for_review = True
                messages.append(_msg(
                    "READY_FOR_REVIEW",
                    "Asset passed automated checks but requires a manual quality review.",
                    _WARN,
                ))
                messages.append(_msg(
                    "VALIDATION_REVIEW_ADVICE",
                    "Check for visible background contamination, UV seams, "
                    "or geometry holes before approving in the dashboard.",
                    _WARN,
                ))
            else:
                # 'fail' or stub — handled at FAILED status but can arrive here
                next_action = "Review the failure details below."
                messages.append(_msg(
                    "READY_FOR_REVIEW",
                    "Asset is validated and awaiting publish decision.",
                    _INFO,
                ))

        elif status == AssetStatus.PUBLISHED:
            next_action = "Asset is live — no action required."
            messages.append(_msg(
                "READY_FOR_PUBLISH",
                "This asset is published and available for AR use.",
                _INFO,
            ))

        elif status == AssetStatus.FAILED:
            next_action = "Review failure details — contact engineering if the issue persists."
            detail = _match_failure_reason(failure_reason or "")
            if detail:
                messages.append(detail)
            else:
                # Generic fallback — do NOT expose raw failure_reason to operator
                messages.append(_msg(
                    "SYSTEM_FAILURE_PIPELINE",
                    "An unexpected pipeline error occurred. "
                    "Contact engineering and provide the session ID.",
                    _CRIT,
                ))

        # ── Step 2: Coverage enrichment ───────────────────────────────────────
        if coverage_report:
            enriched = self._enrich_from_coverage(coverage_report)
            messages.extend(enriched)
            if coverage_report.get("overall_status") != "sufficient":
                should_recapture = True

        # ── Step 3: Validation enrichment ─────────────────────────────────────
        if validation_report:
            enriched = self._enrich_from_validation(validation_report)
            messages.extend(enriched)
            if validation_report.get("final_decision") == "fail":
                should_recapture = True
            elif validation_report.get("final_decision") == "review":
                is_ready_for_review = True

        # ── Step 4: Deduplicate by code (keep first occurrence) ───────────────
        seen: set = set()
        deduped = []
        for m in messages:
            if m["code"] not in seen:
                seen.add(m["code"])
                deduped.append(m)
        messages = deduped

        # ── Step 5: Override next_action for recapture/review states ──────────
        if should_recapture and status not in {AssetStatus.FAILED}:
            next_action = (
                "Re-capture required: follow the operator instructions below "
                "before submitting a new video."
            )
        elif is_ready_for_review and status not in {
            AssetStatus.PUBLISHED, AssetStatus.VALIDATED
        }:
            next_action = "Open the dashboard to review and approve the 3D asset."

        return CaptureGuidance(
            session_id=session_id,
            status=status,
            next_action=next_action,
            should_recapture=should_recapture,
            is_ready_for_review=is_ready_for_review,
            messages=messages,
            coverage_summary=coverage_report,
            validation_summary=validation_report,
        )

    # ── Coverage enrichment ───────────────────────────────────────────────────

    def _enrich_from_coverage(
        self, report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        messages = []

        # ML unavailability — most critical, emit first
        if report.get("ml_segmentation_unavailable"):
            messages.append(_msg(
                "MASKING_DEGRADED_ML",
                "ML segmentation was unavailable during this capture. "
                "Results may contain background contamination. "
                "Install 'rembg' + 'onnxruntime' for full quality.",
                _CRIT,
            ))

        # Per-reason mapping (reasons are human-readable strings from CoverageAnalyzer)
        for reason in report.get("reasons", []):
            r = reason.lower()
            if "unique view" in r or "viewpoint diversity" in r:
                messages.append(_msg(
                    "RECAPTURE_LOW_DIVERSITY",
                    "Not enough unique viewpoints: walk around the object more "
                    "completely, covering at least 4–5 distinct sides.",
                    _CRIT,
                ))
            elif "object motion" in r or "narrow" in r:
                messages.append(_msg(
                    "RECAPTURE_NARROW_MOTION",
                    "Camera barely moved relative to the object. "
                    "Orbit the product in a wide, deliberate arc without stopping.",
                    _WARN,
                ))
            elif "scale" in r or "shape variation" in r:
                messages.append(_msg(
                    "RECAPTURE_SCALE_FLAT",
                    "Shots are all from the same height and distance. "
                    "Include elevated views (30–60°) and vary distance.",
                    _WARN,
                ))
            elif "readable" in r or "few" in r:
                messages.append(_msg(
                    "RECAPTURE_TOO_FEW_FRAMES",
                    "Too few usable frames. Slow down camera motion and ensure "
                    "consistent lighting without motion blur.",
                    _WARN,
                ))
            elif "heuristic fallback" in r or "fallback" in r:
                messages.append(_msg(
                    "RECAPTURE_MASKING_FALLBACK",
                    "Object masks fell back to heuristic mode for too many frames. "
                    "Use a plain, solid-colour background with strong contrast.",
                    _WARN,
                ))
            elif "semantic confidence" in r or "low confidence" in r:
                messages.append(_msg(
                    "RECAPTURE_LOW_CONFIDENCE",
                    "ML object detection confidence was low. "
                    "Improve background contrast, add diffuse lighting, "
                    "avoid reflective or transparent surfaces.",
                    _WARN,
                ))

        # Top-down view advisory (non-blocking)
        if not report.get("top_down_captured", True):
            messages.append(_msg(
                "MISSING_TOP_VIEWS",
                "No elevated-angle shots detected. "
                "Capture 5–10 frames from 30–60° above the product "
                "to capture the top surface for reconstruction.",
                _WARN,
            ))

        return messages

    # ── Validation enrichment ─────────────────────────────────────────────────

    def _enrich_from_validation(
        self, report: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        messages = []
        integrity = report.get("contamination_report", {})

        # Contamination level
        contamination_score = float(report.get("contamination_score", 0.0))
        component_share = float(report.get("largest_component_share", 1.0))

        if contamination_score > 0.5:
            messages.append(_msg(
                "CONTAMINATION_HIGH",
                "The 3D model contains heavy background contamination. "
                "Re-capture in a cleaner area with an unobstructed turntable "
                "or plain surface.",
                _CRIT,
            ))
        elif contamination_score > 0.2:
            messages.append(_msg(
                "CONTAMINATION_HIGH",
                "The 3D model has moderate background contamination. "
                "Ensure the product is isolated on a plain surface "
                "with no items nearby.",
                _WARN,
            ))

        # Component fragmentation
        if component_share < 0.80 and report.get("component_count", 1) > 2:
            messages.append(_msg(
                "CONTAMINATION_FRAGMENTED",
                "The mesh is split into multiple disconnected pieces. "
                "Capture the full product surface without gaps "
                "between angles.",
                _WARN,
            ))

        # Texture / UV
        if integrity.get("texture_uv_integrity") == "fail":
            messages.append(_msg(
                "TEXTURE_UV_FAILURE",
                "Texture mapping failed or produced severe seams. "
                "Improve lighting consistency and avoid over-exposed frames. "
                "Avoid fast panning — keep shots steady.",
                _WARN,
            ))

        # Material semantic status — informational only
        semantic = report.get("material_semantic_status", "")
        if semantic == "diffuse_textured":
            messages.append(_msg(
                "MATERIAL_INFO",
                "Asset is a valid diffuse photoscan — "
                "PBR material maps are not present but geometry+texture is correct.",
                _INFO,
            ))

        # Review state advisory
        if report.get("final_decision") == "review":
            messages.append(_msg(
                "VALIDATION_REVIEW_ADVICE",
                "Automated checks passed with caveats. "
                "Inspect geometry quality, UV seams, and texture coverage "
                "in the 3D viewer before approving.",
                _WARN,
            ))

        return messages

    # ── Markdown rendering ────────────────────────────────────────────────────

    def to_markdown(self, guidance: CaptureGuidance) -> str:
        _icon = {
            GuidanceSeverity.CRITICAL: "🔴 CRITICAL",
            GuidanceSeverity.WARNING:  "🟡 WARNING",
            GuidanceSeverity.INFO:     "ℹ️  INFO",
        }

        lines = [
            f"# Capture Guidance — {guidance.session_id}",
            f"**Status:** `{guidance.status.value.upper()}`",
            f"**Next Action:** {guidance.next_action}",
            "",
            "## Operator Instructions",
        ]

        if not guidance.messages:
            lines.append("- No specific instructions. Processing normally.")
        else:
            for msg in guidance.messages:
                icon = _icon.get(msg["severity"], "—")
                lines.append(f"- {icon}: {msg['message']}")

        lines.append("")

        # Coverage detail
        if guidance.coverage_summary:
            c = guidance.coverage_summary
            lines += [
                "## Coverage Details",
                f"| Metric | Value |",
                f"|---|---|",
                f"| Frames extracted | {c.get('num_frames', '—')} |",
                f"| Readable frames | {c.get('readable_frames', '—')} |",
                f"| Unique viewpoints | {c.get('unique_views', '—')} |",
                f"| Coverage score | {float(c.get('coverage_score', 0.0)):.2f} |",
                f"| Diversity | {c.get('diversity', '—')} |",
                f"| Top-down captured | {c.get('top_down_captured', '—')} |",
                "",
            ]

        # Validation detail
        if guidance.validation_summary:
            v = guidance.validation_summary
            lines += [
                "## Validation Details",
                f"| Metric | Value |",
                f"|---|---|",
                f"| Decision | **{v.get('final_decision', '—').upper()}** |",
                f"| Poly count | {v.get('poly_count', '—')} |",
                f"| Texture status | {v.get('texture_status', '—')} |",
                f"| Contamination score | {float(v.get('contamination_score', 0.0)):.2f} |",
                f"| Component count | {v.get('component_count', '—')} |",
                f"| Mobile grade | {v.get('mobile_performance_grade', '—')} |",
                "",
            ]

        return "\n".join(lines)
