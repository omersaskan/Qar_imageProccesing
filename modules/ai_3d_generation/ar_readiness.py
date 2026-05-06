"""
AR/mobile readiness assessment for AI-generated GLB outputs.

Produces a structured score and verdict from a completed generation manifest.
Does not modify GLB files or perform mesh optimization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Thresholds ─────────────────────────────────────────────────────────────

_FILE_SIZE_GOOD_MB   = 5.0
_FILE_SIZE_REVIEW_MB = 10.0
_VERTEX_COUNT_GOOD   = 50_000
_FACE_COUNT_GOOD     = 100_000
_TEXTURE_GOOD        = 1024
_TEXTURE_REVIEW      = 2048

# Score penalties
_PENALTY_SIZE_LARGE   = 30   # > 10 MB
_PENALTY_SIZE_MEDIUM  = 10   # > 5 MB
_PENALTY_TEXTURE_2048 = 10   # texture_resolution == 2048
_PENALTY_VERTEX_HIGH  = 20   # vertex_count > 50 000
_PENALTY_FACE_HIGH    = 10   # face_count   > 100 000
_PENALTY_REVIEW       = 5    # review_required
_PENALTY_GLB_INVALID  = 40   # GLB failed structural validation
_CAP_PROVIDER_FAILED  = 30   # hard cap when provider_status != ok

# Verdict thresholds
_VERDICT_MOBILE_READY_MIN = 80
_VERDICT_REVIEW_MIN       = 50


def assess_ar_readiness(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess AR/mobile readiness of a generation manifest.

    Parameters
    ----------
    manifest : dict
        Completed ai_3d_generation manifest dict (as returned by generate_ai_3d).

    Returns
    -------
    dict with keys: enabled, score, verdict, checks, warnings, recommendations
    """
    warnings: List[str]       = []
    recommendations: List[str] = []
    score = 100

    # ── Extract fields ────────────────────────────────────────────────────────
    output_glb_path    = manifest.get("output_glb_path")
    output_size_bytes  = manifest.get("output_size_bytes") or 0
    provider_status    = manifest.get("provider_status", "failed")
    quality_gate       = manifest.get("quality_gate") or {}
    quality_gate_verdict = quality_gate.get("verdict", "failed")
    worker_meta        = manifest.get("worker_metadata") or {}
    texture_resolution = worker_meta.get("texture_resolution")
    review_required    = manifest.get("review_required", True)

    # Vertex/face: prefer top-level manifest fields; fall back to manifest["mesh_stats"]
    # populated by the pipeline via extract_mesh_stats().
    _mesh_stats = manifest.get("mesh_stats") or {}
    _vc = manifest.get("vertex_count")
    vertex_count: Optional[int] = _vc if _vc is not None else _mesh_stats.get("vertex_count")
    _fc = manifest.get("face_count")
    face_count:   Optional[int] = _fc if _fc is not None else _mesh_stats.get("face_count")

    # GLB validation: None = not run, True = passed, False = failed
    _glb_val = manifest.get("glb_validation") or {}
    glb_valid: Optional[bool] = _glb_val.get("valid")  # None when absent

    # ── Check: GLB exists ─────────────────────────────────────────────────────
    glb_exists = bool(output_glb_path and Path(output_glb_path).exists())
    if not glb_exists:
        return {
            "enabled": True,
            "score": 0,
            "verdict": "not_ready",
            "checks": {
                "glb_exists":          {"ok": False, "value": output_glb_path},
                "file_size_mb":        {"ok": False, "value": None},
                "vertex_count":        {"ok": None,  "value": None},
                "face_count":          {"ok": None,  "value": None},
                "texture_resolution":  {"ok": None,  "value": texture_resolution},
                "provider_status":     {"ok": False, "value": provider_status},
                "quality_gate":        {"ok": False, "value": quality_gate_verdict},
            },
            "warnings": ["glb_output_missing"],
            "recommendations": ["Regenerate — no GLB output was produced."],
        }

    # ── Check: provider_status ────────────────────────────────────────────────
    provider_ok = provider_status == "ok"
    if not provider_ok:
        score = min(score, _CAP_PROVIDER_FAILED)
        warnings.append("provider_status_not_ok")
        recommendations.append("Provider did not report success; GLB may be incomplete.")

    # ── Check: file size ──────────────────────────────────────────────────────
    file_size_mb = round(output_size_bytes / (1024 * 1024), 3) if output_size_bytes else 0.0
    if file_size_mb > _FILE_SIZE_REVIEW_MB:
        score -= _PENALTY_SIZE_LARGE
        warnings.append("file_size_too_large")
        recommendations.append(
            f"GLB is {file_size_mb:.1f} MB (>{_FILE_SIZE_REVIEW_MB} MB). "
            "Consider mesh optimization before AR delivery."
        )
        size_ok = False
    elif file_size_mb > _FILE_SIZE_GOOD_MB:
        score -= _PENALTY_SIZE_MEDIUM
        warnings.append("file_size_large")
        recommendations.append(
            f"GLB is {file_size_mb:.1f} MB (>{_FILE_SIZE_GOOD_MB} MB). "
            "May be heavy for low-end mobile AR."
        )
        size_ok = False
    else:
        size_ok = True

    # ── Check: texture_resolution ─────────────────────────────────────────────
    if texture_resolution is not None:
        texture_ok = texture_resolution <= _TEXTURE_GOOD
        if texture_resolution >= _TEXTURE_REVIEW:
            score -= _PENALTY_TEXTURE_2048
            warnings.append("texture_resolution_2048")
            recommendations.append(
                "Ultra mode uses 2048px textures. Consider 'high' mode for mobile delivery."
            )
    else:
        texture_ok = True  # unknown — don't penalise

    # ── Check: vertex_count ───────────────────────────────────────────────────
    if vertex_count is not None:
        vertex_ok = vertex_count <= _VERTEX_COUNT_GOOD
        if not vertex_ok:
            score -= _PENALTY_VERTEX_HIGH
            warnings.append("vertex_count_high")
            recommendations.append(
                f"Vertex count {vertex_count:,} exceeds {_VERTEX_COUNT_GOOD:,}. "
                "May impact mobile frame rate."
            )
    else:
        vertex_ok = None

    # ── Check: face_count ─────────────────────────────────────────────────────
    if face_count is not None:
        face_ok = face_count <= _FACE_COUNT_GOOD
        if not face_ok:
            score -= _PENALTY_FACE_HIGH
            warnings.append("face_count_high")
            recommendations.append(
                f"Face count {face_count:,} exceeds {_FACE_COUNT_GOOD:,}. "
                "May impact mobile frame rate."
            )
    else:
        face_ok = None

    # ── Check: quality_gate ───────────────────────────────────────────────────
    gate_ok = quality_gate_verdict in ("ok", "review")

    # ── Check: GLB validation ─────────────────────────────────────────────────
    if glb_valid is False:
        score -= _PENALTY_GLB_INVALID
        warnings.append("glb_validation_failed")
        recommendations.append(
            "GLB failed structural validation. Inspect issues before AR delivery."
        )

    # ── Review penalty ────────────────────────────────────────────────────────
    if review_required:
        score -= _PENALTY_REVIEW

    # ── Clamp ─────────────────────────────────────────────────────────────────
    score = max(0, min(100, score))

    # ── Verdict ───────────────────────────────────────────────────────────────
    # glb_valid=False (failed validation) blocks mobile_ready regardless of score
    required_ok = glb_exists and provider_ok and (glb_valid is not False)
    if required_ok and score >= _VERDICT_MOBILE_READY_MIN:
        verdict = "mobile_ready"
    elif score >= _VERDICT_REVIEW_MIN:
        verdict = "review"
    else:
        verdict = "not_ready"

    return {
        "enabled": True,
        "score": score,
        "verdict": verdict,
        "checks": {
            "glb_exists":         {"ok": glb_exists,              "value": output_glb_path},
            "file_size_mb":       {"ok": size_ok,                  "value": file_size_mb},
            "vertex_count":       {"ok": vertex_ok,                "value": vertex_count},
            "face_count":         {"ok": face_ok,                  "value": face_count},
            "texture_resolution": {"ok": texture_ok,               "value": texture_resolution},
            "provider_status":    {"ok": provider_ok,              "value": provider_status},
            "quality_gate":       {"ok": gate_ok,                  "value": quality_gate_verdict},
            "glb_validation":     {"ok": None if glb_valid is None else glb_valid,
                                   "value": glb_valid},
        },
        "warnings": warnings,
        "recommendations": recommendations,
    }
