"""
Cleanup report generator — writes cleanup_report.json and cleanup_report.md.

write_cleanup_report(output_dir, manifest, asset_quality, mesh_cleanup,
                     normalization, pbr_textures, export_profiles) -> dict
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_CLEANUP_REPORT_ENABLED = (
    os.environ.get("AI_3D_CLEANUP_REPORT_ENABLED", "true").lower() == "true"
)


def write_cleanup_report(
    output_dir: Optional[str],
    manifest: Dict[str, Any],
    asset_quality: Dict[str, Any],
    mesh_cleanup: Dict[str, Any],
    normalization: Dict[str, Any],
    pbr_textures: Dict[str, Any],
    export_profiles: Dict[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "enabled": _CLEANUP_REPORT_ENABLED,
        "available": False,
        "json_path": None,
        "markdown_path": None,
        "issue_count": 0,
        "blocking_issue_count": 0,
        "manual_cleanup_required": False,
        "retopology_recommended": False,
        "warnings": [],
        "error": None,
    }

    if not _CLEANUP_REPORT_ENABLED:
        return result

    if not output_dir:
        result["warnings"].append("output_dir_missing")
        return result

    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        mc_issues: List[str] = mesh_cleanup.get("issues") or []
        mc_warnings: List[str] = mesh_cleanup.get("warnings") or []
        mc_metrics: Dict[str, Any] = mesh_cleanup.get("metrics") or {}
        mc_recs: List[str] = mesh_cleanup.get("recommendations") or []

        norm_issues: List[str] = normalization.get("issues") or []
        norm_warnings: List[str] = normalization.get("warnings") or []
        norm_analysis: Dict[str, Any] = normalization.get("analysis") or {}

        pbr_issues: List[str] = pbr_textures.get("issues") or []
        pbr_warnings: List[str] = pbr_textures.get("warnings") or []

        mobile_blockers: List[str] = (export_profiles.get("mobile_ar") or {}).get("blocking_reasons") or []

        session_id = manifest.get("session_id", "unknown")
        aq_verdict = asset_quality.get("verdict", "unknown")
        aq_score = asset_quality.get("score")

        floating = "floating_parts_detected" in mc_warnings
        non_manifold = "non_manifold_geometry" in mc_warnings
        high_component = any(w.startswith("high_component_count") for w in mc_warnings)
        component_count = mc_metrics.get("component_count")
        degenerate_count = mc_metrics.get("degenerate_face_count") or 0

        ground_uncertain = "ground_alignment_uncertain" in norm_warnings
        not_centered = "model_not_centered" in norm_warnings

        manual_cleanup_required = bool(
            mc_issues or floating or non_manifold or high_component or norm_issues
        )
        retopology_recommended = bool(
            non_manifold or high_component or "retopology_recommended" in mc_recs
        )

        # Signal-based issue count (not double-counting)
        signal_flags = [floating, non_manifold, high_component, ground_uncertain, not_centered]
        issue_count = len(mc_issues) + len(norm_issues) + len(pbr_issues) + sum(signal_flags)
        blocking_issue_count = len(mobile_blockers)

        report = {
            "session_id": session_id,
            "asset_quality_verdict": aq_verdict,
            "asset_quality_score": aq_score,
            "summary": {
                "issue_count": issue_count,
                "blocking_issue_count": blocking_issue_count,
                "manual_cleanup_required": manual_cleanup_required,
                "retopology_recommended": retopology_recommended,
            },
            "mesh_issues": {
                "cleanup_status": mesh_cleanup.get("status", "ok"),
                "component_count": component_count,
                "floating_parts_detected": floating,
                "non_manifold_geometry": non_manifold,
                "degenerate_faces": degenerate_count,
                "high_component_count": high_component,
                "issues": mc_issues,
                "warnings": mc_warnings,
            },
            "normalization_issues": {
                "ground_alignment_uncertain": ground_uncertain,
                "model_not_centered": not_centered,
                "likely_flat_on_ground": norm_analysis.get("likely_flat_on_ground"),
                "ground_offset": norm_analysis.get("ground_offset"),
                "issues": norm_issues,
                "warnings": norm_warnings,
            },
            "pbr_issues": {
                "material_count": pbr_textures.get("material_count", 0),
                "texture_count": pbr_textures.get("texture_count", 0),
                "has_base_color": pbr_textures.get("has_base_color"),
                "issues": pbr_issues,
                "warnings": pbr_warnings,
            },
            "delivery_blockers": mobile_blockers,
            "recommended_manual_steps": _manual_steps(
                floating, non_manifold, high_component, component_count,
                degenerate_count, ground_uncertain, not_centered, pbr_issues,
            ),
            "recommended_automated_future_steps": _automated_steps(
                retopology_recommended, ground_uncertain, not_centered, degenerate_count,
            ),
        }

        json_path = Path(output_dir) / "cleanup_report.json"
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        md_path = Path(output_dir) / "cleanup_report.md"
        md_path.write_text(_to_markdown(report), encoding="utf-8")

        result.update({
            "available": True,
            "json_path": "cleanup_report.json",
            "markdown_path": "cleanup_report.md",
            "issue_count": issue_count,
            "blocking_issue_count": blocking_issue_count,
            "manual_cleanup_required": manual_cleanup_required,
            "retopology_recommended": retopology_recommended,
        })

    except Exception as exc:
        msg = str(exc)
        for sep in ("\\", "/"):
            if sep in msg:
                msg = msg.split(sep)[-1]
        result["error"] = msg[:200]
        result["warnings"].append("cleanup_report_failed")
        log.warning("write_cleanup_report failed: %s", exc)

    return result


def _manual_steps(
    floating: bool,
    non_manifold: bool,
    high_component: bool,
    component_count: Optional[int],
    degenerate_count: int,
    ground_uncertain: bool,
    not_centered: bool,
    pbr_issues: List[str],
) -> List[str]:
    steps = []
    if floating:
        steps.append("Inspect and remove floating/disconnected mesh components.")
    if non_manifold:
        steps.append("Fix non-manifold geometry — retopology may be required.")
    if high_component:
        n = f" ({component_count})" if component_count else ""
        steps.append(f"Review and merge/clean the separate mesh components{n}.")
    if degenerate_count > 0:
        steps.append(f"Remove {degenerate_count} degenerate (zero-area) face(s).")
    if ground_uncertain:
        steps.append("Align model to ground plane (Y=0) before AR/mobile delivery.")
    if not_centered:
        steps.append("Center model at origin for AR/web delivery.")
    if "no_materials" in pbr_issues:
        steps.append("Add materials and textures for production delivery.")
    return steps


def _automated_steps(
    retopology_recommended: bool,
    ground_uncertain: bool,
    not_centered: bool,
    degenerate_count: int,
) -> List[str]:
    steps = []
    if retopology_recommended:
        steps.append(
            "Run retopology (e.g. ZRemesher, InstantMesh) after manual component cleanup."
        )
    if ground_uncertain or not_centered:
        steps.append(
            "Apply automated normalization transforms via AQ2 normalized copy."
        )
    if degenerate_count > 100:
        steps.append("Run automated degenerate face removal (mesh repair tool).")
    return steps


def _to_markdown(r: Dict[str, Any]) -> str:
    s = r["summary"]
    mi = r["mesh_issues"]
    ni = r["normalization_issues"]
    pi = r["pbr_issues"]

    def _yn(v: Any) -> str:
        if v is None:
            return "—"
        return "✓ Yes" if v else "✗ No"

    lines = [
        "# Asset Quality Cleanup Report",
        "",
        f"**Session:** {r['session_id']}  ",
        f"**Verdict:** {r['asset_quality_verdict']}"
        + (f" (score: {r['asset_quality_score']})" if r['asset_quality_score'] is not None else ""),
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total issues | {s['issue_count']} |",
        f"| Blocking issues | {s['blocking_issue_count']} |",
        f"| Manual cleanup required | {_yn(s['manual_cleanup_required'])} |",
        f"| Retopology recommended | {_yn(s['retopology_recommended'])} |",
        "",
        "## Mesh Issues",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Cleanup status | {mi['cleanup_status']} |",
        f"| Component count | {mi['component_count'] or '—'} |",
        f"| Floating parts | {_yn(mi['floating_parts_detected'])} |",
        f"| Non-manifold geometry | {_yn(mi['non_manifold_geometry'])} |",
        f"| Degenerate faces | {mi['degenerate_faces']} |",
        f"| High component count | {_yn(mi['high_component_count'])} |",
        "",
        "## Normalization Issues",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Ground alignment | {'✗ Uncertain' if ni['ground_alignment_uncertain'] else '✓ OK'}"
        + (f" (offset: {ni['ground_offset']:.3f})" if ni.get('ground_offset') is not None else "") + " |",
        f"| Model centered | {'✗ Off-center' if ni['model_not_centered'] else '✓ OK'} |",
        f"| Likely flat on ground | {_yn(ni['likely_flat_on_ground'])} |",
        "",
        "## PBR / Material Issues",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Materials | {pi['material_count']} |",
        f"| Textures | {pi['texture_count']} |",
        f"| Has base color | {_yn(pi['has_base_color'])} |",
    ]

    if r["delivery_blockers"]:
        lines += ["", "## Delivery Blockers", ""]
        for b in r["delivery_blockers"]:
            lines.append(f"- `{b}`")

    if r["recommended_manual_steps"]:
        lines += ["", "## Recommended Manual Steps", ""]
        for i, step in enumerate(r["recommended_manual_steps"], 1):
            lines.append(f"{i}. {step}")

    if r["recommended_automated_future_steps"]:
        lines += ["", "## Recommended Automated Future Steps", ""]
        for i, step in enumerate(r["recommended_automated_future_steps"], 1):
            lines.append(f"{i}. {step}")

    return "\n".join(lines) + "\n"
