"""
Export profile readiness assessment.

assess_export_profiles(manifest, asset_quality) -> dict

Profiles: raw, web_preview, mobile_ar, desktop_high
Uses existing glb_validation, mesh_stats, ar_readiness, and asset_quality checks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def assess_export_profiles(
    manifest: Dict[str, Any],
    asset_quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assess readiness for each export profile.

    Parameters
    ----------
    manifest : dict
        Completed AI 3D generation manifest.
    asset_quality : dict | None
        Preliminary asset_quality result (may be partial during pipeline build).

    Returns
    -------
    dict with keys: raw, web_preview, mobile_ar, desktop_high
    """
    output_glb_path = manifest.get("output_glb_path")
    glb_validation = manifest.get("glb_validation") or {}
    glb_valid: Optional[bool] = glb_validation.get("valid")
    mesh_stats = manifest.get("mesh_stats") or {}
    ar_readiness = manifest.get("ar_readiness") or {}

    aq = asset_quality or {}
    aq_checks = aq.get("checks") or {}
    mesh_cleanup = aq_checks.get("mesh_cleanup") or {}
    pbr = aq_checks.get("pbr_textures") or {}

    face_count: Optional[int] = mesh_stats.get("face_count")
    output_size_bytes: int = manifest.get("output_size_bytes") or 0
    file_size_mb = output_size_bytes / (1024 * 1024) if output_size_bytes else 0.0

    glb_exists = bool(output_glb_path and Path(output_glb_path).exists())
    cleanup_status = mesh_cleanup.get("status", "ok")
    pbr_issues: List[str] = pbr.get("issues") or []
    ar_verdict = ar_readiness.get("verdict", "")

    # ── Raw ───────────────────────────────────────────────────────────────────
    raw = {
        "available": glb_exists,
        "path": None,  # never expose server-side filesystem paths in API responses
        "valid": glb_valid if glb_valid is not None else glb_exists,
    }

    # ── Web Preview ───────────────────────────────────────────────────────────
    web_blocking: List[str] = []
    web_recs: List[str] = []
    if not glb_exists:
        web_blocking.append("glb_missing")
    if glb_valid is False:
        web_blocking.append("glb_validation_failed")
    if file_size_mb > 50:
        web_blocking.append("file_size_too_large_for_web")
    if face_count and face_count > 1_000_000:
        web_blocking.append("face_count_too_high_for_web")
    if not web_blocking and face_count and face_count > 200_000:
        web_recs.append("Consider mesh decimation for better web loading performance.")

    web_preview = {
        "ready": len(web_blocking) == 0,
        "blocking_reasons": web_blocking,
        "recommendations": web_recs,
    }

    # ── Mobile AR ─────────────────────────────────────────────────────────────
    mobile_blocking: List[str] = []
    mobile_recs: List[str] = []
    if not glb_exists:
        mobile_blocking.append("glb_missing")
    if glb_valid is False:
        mobile_blocking.append("glb_validation_failed")
    if file_size_mb > 10:
        mobile_blocking.append("file_size_too_large_for_mobile")
    if face_count and face_count > 100_000:
        mobile_blocking.append("face_count_too_high_for_mobile")
    if ar_verdict == "not_ready":
        mobile_blocking.append("ar_readiness_not_ready")
    if cleanup_status == "failed":
        mobile_blocking.append("mesh_cleanup_failed")
    if "no_materials" in pbr_issues:
        mobile_recs.append(
            "No materials found — model will render untextured on AR platforms."
        )

    mobile_ar = {
        "ready": len(mobile_blocking) == 0,
        "blocking_reasons": mobile_blocking,
        "recommendations": mobile_recs,
    }

    # ── Desktop High ──────────────────────────────────────────────────────────
    desktop_blocking: List[str] = []
    desktop_recs: List[str] = []
    if not glb_exists:
        desktop_blocking.append("glb_missing")
    if glb_valid is False:
        desktop_blocking.append("glb_validation_failed")
    if cleanup_status == "failed":
        desktop_recs.append(
            "Mesh cleanup issues detected — review before desktop delivery."
        )

    desktop_high = {
        "ready": len(desktop_blocking) == 0,
        "blocking_reasons": desktop_blocking,
        "recommendations": desktop_recs,
    }

    return {
        "raw": raw,
        "web_preview": web_preview,
        "mobile_ar": mobile_ar,
        "desktop_high": desktop_high,
    }
