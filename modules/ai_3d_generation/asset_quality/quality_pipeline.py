"""
Provider-neutral asset quality pipeline orchestrator.

run_asset_quality_pipeline(glb_path, manifest) -> dict

Runs after glb_validation, mesh_stats, and ar_readiness are populated
in the manifest. Does not modify the GLB file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .normalization import analyze_normalization
from .mesh_cleanup_audit import audit_mesh_cleanup
from .lod import build_lod_plan
from .pbr_audit import audit_pbr_textures
from .export_profiles import assess_export_profiles

log = logging.getLogger(__name__)

# Score thresholds
_SCORE_PRODUCTION_READY = 85
_SCORE_MOBILE_READY = 65
_SCORE_NEEDS_REVIEW = 40

# Score penalties
_PENALTY_GLB_INVALID = 30
_PENALTY_MESH_FAILED = 25
_PENALTY_MESH_REVIEW = 10
_PENALTY_FLOATING_PARTS = 10
_PENALTY_DEGENERATE_FACES = 5
_PENALTY_HIGH_FACE_COUNT = 10
_PENALTY_NO_MATERIALS = 15
_PENALTY_MISSING_BASE_COLOR = 10
_PENALTY_NO_TEXTURE_WITH_MATERIALS = 5


def run_asset_quality_pipeline(
    glb_path: Optional[str],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the full asset quality audit pipeline.

    Returns asset_quality dict with all sub-checks.
    Never raises — on exception returns a degraded result with available=False.
    """
    try:
        return _run_pipeline(glb_path, manifest)
    except Exception as exc:
        log.warning("asset_quality_pipeline failed: %s", exc)
        return {
            "enabled": True,
            "available": False,
            "status": "review",
            "score": None,
            "verdict": "needs_review",
            "provider_neutral": True,
            "checks": {
                "scale_orientation": {"enabled": True, "available": False},
                "mesh_cleanup": {"enabled": True, "available": False},
                "lod": {"enabled": True, "available": False},
                "pbr_textures": {"enabled": True, "available": False},
                "export_profiles": {},
            },
            "warnings": ["asset_quality_pipeline_failed"],
            "recommendations": [],
            "error": _sanitize_error(exc),
        }


def _sanitize_error(exc: Exception) -> str:
    msg = str(exc)
    for sep in ("\\", "/"):
        if sep in msg:
            parts = msg.split(sep)
            msg = parts[-1]
    return msg[:200]


def _run_pipeline(
    glb_path: Optional[str],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    warnings: List[str] = []
    recommendations: List[str] = []
    score = 100

    glb_exists = bool(glb_path and Path(glb_path).exists())
    glb_validation = manifest.get("glb_validation") or {}
    glb_valid: Optional[bool] = glb_validation.get("valid")
    mesh_stats = manifest.get("mesh_stats") or {}
    ar_readiness = manifest.get("ar_readiness") or {}

    # ── Sub-audits ────────────────────────────────────────────────────────────
    normalization = analyze_normalization(glb_path)

    mesh_cleanup = audit_mesh_cleanup(glb_path)

    output_dir: Optional[str] = None
    if glb_path:
        try:
            output_dir = str(Path(glb_path).parent)
        except Exception:
            pass

    lod = build_lod_plan(
        mesh_stats=mesh_stats,
        ar_readiness=ar_readiness,
        asset_quality_context={"glb_path": glb_path, "output_dir": output_dir},
    )

    pbr_textures = audit_pbr_textures(glb_path)

    # Preliminary assembly for export profiles
    _prelim: Dict[str, Any] = {
        "checks": {
            "mesh_cleanup": mesh_cleanup,
            "pbr_textures": pbr_textures,
        }
    }
    export_profiles = assess_export_profiles(manifest, _prelim)

    # ── Scoring ───────────────────────────────────────────────────────────────
    cleanup_status = mesh_cleanup.get("status", "ok")

    if not glb_exists:
        score = 0
    else:
        if glb_valid is False:
            score -= _PENALTY_GLB_INVALID
            warnings.append("glb_validation_failed")

        if cleanup_status == "failed":
            score -= _PENALTY_MESH_FAILED
            warnings.append("mesh_cleanup_failed")
            recommendations.append(
                "Mesh has critical quality issues. Manual cleanup required."
            )
        elif cleanup_status == "review":
            score -= _PENALTY_MESH_REVIEW
            warnings.append("mesh_cleanup_review")

        if "floating_parts_detected" in (mesh_cleanup.get("warnings") or []):
            score -= _PENALTY_FLOATING_PARTS
            warnings.append("floating_parts_detected")

        degen = (mesh_cleanup.get("metrics") or {}).get("degenerate_face_count") or 0
        if degen > 100:
            score -= _PENALTY_DEGENERATE_FACES
            warnings.append("degenerate_faces_detected")

        face_count = mesh_stats.get("face_count")
        if face_count and face_count > 200_000:
            score -= _PENALTY_HIGH_FACE_COUNT
            recommendations.append("High face count — LOD generation recommended.")

        pbr_issues = pbr_textures.get("issues") or []
        if "no_materials" in pbr_issues:
            score -= _PENALTY_NO_MATERIALS
            warnings.append("pbr_review_required")
        elif pbr_textures.get("has_base_color") is False:
            score -= _PENALTY_MISSING_BASE_COLOR
            warnings.append("pbr_review_required")

        if (
            pbr_textures.get("texture_count", 0) == 0
            and pbr_textures.get("material_count", 0) > 0
        ):
            score -= _PENALTY_NO_TEXTURE_WITH_MATERIALS

    score = max(0, min(100, score))

    # ── Verdict (conservative) ────────────────────────────────────────────────
    severe = not glb_exists or glb_valid is False or cleanup_status == "failed"

    if not glb_exists:
        verdict = "not_ready"
    elif severe:
        verdict = "needs_review"
    elif score >= _SCORE_PRODUCTION_READY and cleanup_status == "ok":
        verdict = "production_ready"
    elif score >= _SCORE_MOBILE_READY:
        verdict = "mobile_ready"
    elif score >= _SCORE_NEEDS_REVIEW:
        verdict = "needs_review"
    else:
        verdict = "not_ready"

    # AR readiness downgrade guard: mobile_ready requires AR also ready
    ar_verdict = ar_readiness.get("verdict", "")
    if ar_verdict == "not_ready" and verdict == "mobile_ready":
        verdict = "needs_review"
        warnings.append("ar_readiness_not_ready_downgrade")

    # ── Final status ──────────────────────────────────────────────────────────
    if verdict == "not_ready" or not glb_exists:
        status = "failed"
    elif verdict in ("needs_review",) or warnings:
        status = "review"
    else:
        status = "ok"

    return {
        "enabled": True,
        "available": True,
        "status": status,
        "score": score,
        "verdict": verdict,
        "provider_neutral": True,
        "checks": {
            "scale_orientation": normalization,
            "mesh_cleanup": mesh_cleanup,
            "lod": lod,
            "pbr_textures": pbr_textures,
            "export_profiles": export_profiles,
        },
        "warnings": warnings,
        "recommendations": recommendations,
        "error": None,
    }
