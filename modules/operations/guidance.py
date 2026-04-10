from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from modules.shared_contracts.models import (
    CaptureGuidance, GuidanceSeverity, AssetStatus
)

class GuidanceAggregator:
    """
    Assembles operator-facing guidance by interpreting backend reports 
    (coverage, validation) and current session status.
    """

    def generate_guidance(
        self, 
        session_id: str, 
        status: AssetStatus,
        coverage_report: Optional[Dict[str, Any]] = None,
        validation_report: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None
    ) -> CaptureGuidance:
        
        messages = []
        should_recapture = False
        is_ready_for_review = False
        next_action = "Wait for processing..."

        # 1. Map Status to Base Logic
        if status == AssetStatus.CREATED:
            next_action = "Upload video and start capture processing."
            messages.append({
                "code": "AWAITING_UPLOAD",
                "message": "Waiting for video upload to begin extraction.",
                "severity": GuidanceSeverity.INFO
            })
        
        elif status == AssetStatus.CAPTURED:
            next_action = "Geometry reconstruction in progress. No action needed."
            messages.append({
                "code": "PROCESSING_RECONSTRUCTION",
                "message": "Backend is now reconstructing 3D geometry from frames.",
                "severity": GuidanceSeverity.INFO
            })

        elif status == AssetStatus.RECAPTURE_REQUIRED:
            next_action = "Analyze the issues below and retake the video."
            should_recapture = True
            messages.append({
                "code": "QUALITY_BAR_NOT_MET",
                "message": "The previous capture did not meet production quality standards.",
                "severity": GuidanceSeverity.CRITICAL
            })
            if failure_reason:
                 messages.append({
                    "code": "FAILURE_REASON",
                    "message": f"Specific issue: {failure_reason}",
                    "severity": GuidanceSeverity.WARNING
                })

        elif status == AssetStatus.VALIDATED:
            next_action = "Review the generated asset for publication."
            is_ready_for_review = True
            messages.append({
                "code": "READY_FOR_REVIEW",
                "message": "Asset is validated. Please perform a final visual check.",
                "severity": GuidanceSeverity.INFO
            })

        elif status == AssetStatus.FAILED:
            next_action = "Contact engineering or restart the capture."
            messages.append({
                "code": "SYSTEM_FAILURE",
                "message": f"Pipeline failed: {failure_reason or 'Unknown error'}",
                "severity": GuidanceSeverity.CRITICAL
            })

        # 2. Enrich with Coverage Data
        if coverage_report:
            self._enrich_from_coverage(coverage_report, messages)
            if coverage_report.get("overall_status") != "sufficient":
                should_recapture = True

        # 3. Enrich with Validation Data
        if validation_report:
            self._enrich_from_validation(validation_report, messages)
            if validation_report.get("final_decision") == "fail":
                should_recapture = True
            elif validation_report.get("final_decision") == "review":
                 is_ready_for_review = True

        # 4. Final Next Action Polishing
        if should_recapture:
            next_action = "RECUT/RETAKE: Follow the guidance below to improve the next capture."
        elif is_ready_for_review:
            next_action = "REVIEW: Open the dashboard to inspect the 3D model."

        return CaptureGuidance(
            session_id=session_id,
            status=status,
            next_action=next_action,
            should_recapture=should_recapture,
            is_ready_for_review=is_ready_for_review,
            messages=messages,
            coverage_summary=coverage_report,
            validation_summary=validation_report
        )

    def _enrich_from_coverage(self, report: Dict[str, Any], messages: List[Dict[str, Any]]):
        diversity = report.get("diversity", "unknown")
        if diversity == "insufficient":
            messages.append({
                "code": "LOW_DIVERSITY",
                "message": "Rotate the object more horizontally or capture from more unique side angles.",
                "severity": GuidanceSeverity.WARNING
            })

        if not report.get("top_down_captured", True):
            messages.append({
                "code": "MISSING_TOP_VIEWS",
                "message": "Capture more frames from an elevated position looking down at the object.",
                "severity": GuidanceSeverity.WARNING
            })

        reasons = report.get("reasons", [])
        for reason in reasons:
            if "low semantic confidence" in reason.lower() or "fallback" in reason.lower():
                messages.append({
                    "code": "MASKING_QUALITY",
                    "message": "Object masking is struggling. Ensure a cleaner background and good contrast.",
                    "severity": GuidanceSeverity.WARNING
                })

    def _enrich_from_validation(self, report: Dict[str, Any], messages: List[Dict[str, Any]]):
        contamination = report.get("contamination_score", 0.0)
        if contamination > 0.3:
            messages.append({
                "code": "CONTAMINATION",
                "message": "The 3D model contains significant environment clutter. Retake in a clearer spot.",
                "severity": GuidanceSeverity.WARNING
            })

        integrity_report = report.get("contamination_report", {})
        if integrity_report.get("texture_uv_integrity") == "fail":
            messages.append({
                "code": "TEXTURE_FAILURE",
                "message": "Texture mapping failed or degraded. Check lighting and avoid blurry motion.",
                "severity": GuidanceSeverity.CRITICAL
            })
            
        semantic_status = report.get("material_semantic_status", "unknown")
        if semantic_status == "diffuse_textured":
            messages.append({
                "code": "SEMANTIC_INFO",
                "message": "Asset is a valid photogrammetry scan (Diffuse only).",
                "severity": GuidanceSeverity.INFO
            })

    def to_markdown(self, guidance: CaptureGuidance) -> str:
        lines = []
        lines.append(f"# Capture Guidance: {guidance.session_id}")
        lines.append(f"**Current Status:** `{guidance.status.value.upper()}`")
        lines.append(f"**Next Action:** {guidance.next_action}")
        lines.append("")
        
        lines.append("## Operator Instructions")
        if not guidance.messages:
            lines.append("- No specific instructions. Processing normally.")
        else:
            for msg in guidance.messages:
                prefix = "🔴 CRITICAL:" if msg["severity"] == GuidanceSeverity.CRITICAL else \
                         "🟡 WARNING:" if msg["severity"] == GuidanceSeverity.WARNING else "ℹ️ INFO:"
                lines.append(f"- {prefix} {msg['message']}")
        
        lines.append("")
        if guidance.coverage_summary:
            lines.append("## Coverage Details")
            c = guidance.coverage_summary
            lines.append(f"- Unique Views: {c.get('unique_views', 0)}")
            lines.append(f"- Overall Score: {float(c.get('coverage_score', 0.0)):.2f}")
            lines.append(f"- Diversity: {c.get('diversity', 'N/A')}")
        
        return "\n".join(lines)
