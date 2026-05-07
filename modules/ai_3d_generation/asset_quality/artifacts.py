"""
AQ2 artifact pipeline orchestrator.

run_aq2_pipeline(raw_glb_path, session_dir, manifest, asset_quality) -> dict

Produces:
- normalized_copy  (normalized.glb)
- cleanup_report   (cleanup_report.json + .md)
- export_package   (export_package/ folder)

Never overwrites the raw GLB. Never performs destructive cleanup.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .normalized_copy import create_normalized_copy
from .cleanup_report import write_cleanup_report
from .export_package import create_export_package

log = logging.getLogger(__name__)

_AQ2_ENABLED = os.environ.get("AI_3D_AQ2_ENABLED", "true").lower() == "true"
_DESTRUCTIVE_CLEANUP_ENABLED = False   # always off, not configurable via env
_OVERWRITE_RAW = False                  # always off, not configurable via env


def run_aq2_pipeline(
    raw_glb_path: Optional[str],
    session_dir: Optional[str],
    manifest: Dict[str, Any],
    asset_quality: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Orchestrate AQ2 artifacts: normalized copy -> cleanup report -> export package.
    Never raises — degrades gracefully on failure.
    """
    result: Dict[str, Any] = {
        "enabled": _AQ2_ENABLED,
        "status": "ok",
        "normalized_copy": {},
        "cleanup_report": {},
        "export_package": {},
        "warnings": [],
        "error": None,
    }

    if not _AQ2_ENABLED:
        return result

    try:
        # Extract AQ1 sub-checks from asset_quality or fall back to manifest top-level
        aq_checks = asset_quality.get("checks") or {}
        normalization = aq_checks.get("scale_orientation") or manifest.get("normalization") or {}
        mesh_cleanup = aq_checks.get("mesh_cleanup") or manifest.get("mesh_cleanup") or {}
        pbr_textures = aq_checks.get("pbr_textures") or manifest.get("pbr_textures") or {}
        export_profiles = manifest.get("export_profiles") or {}

        # Step 1 — normalized copy
        normalized_copy = create_normalized_copy(
            raw_glb_path=raw_glb_path,
            output_dir=session_dir,
            normalization_analysis=normalization,
        )
        result["normalized_copy"] = normalized_copy

        # Step 2 — cleanup report
        cleanup_report = write_cleanup_report(
            output_dir=session_dir,
            manifest=manifest,
            asset_quality=asset_quality,
            mesh_cleanup=mesh_cleanup,
            normalization=normalization,
            pbr_textures=pbr_textures,
            export_profiles=export_profiles,
        )
        result["cleanup_report"] = cleanup_report

        # Step 3 — export package
        export_package = create_export_package(
            session_dir=session_dir,
            raw_glb_path=raw_glb_path,
            normalized_copy=normalized_copy,
            cleanup_report=cleanup_report,
            asset_quality=asset_quality,
            export_profiles=export_profiles,
        )
        result["export_package"] = export_package

        any_error = any([
            not normalized_copy.get("enabled"),
            cleanup_report.get("error"),
            export_package.get("error"),
        ])
        result["status"] = "review" if any_error else "ok"

    except Exception as exc:
        msg = str(exc)
        for sep in ("\\", "/"):
            if sep in msg:
                msg = msg.split(sep)[-1]
        result["error"] = msg[:200]
        result["status"] = "review"
        result["warnings"].append("aq2_pipeline_failed")
        log.warning("run_aq2_pipeline failed: %s", exc)

    return result


def update_export_profiles_recommended_artifact(
    export_profiles: Dict[str, Any],
    normalized_copy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Annotate each export profile with recommended_artifact (raw|normalized|none).
    Modifies and returns export_profiles.
    """
    norm_valid = (
        normalized_copy.get("available")
        and (normalized_copy.get("validation") or {}).get("valid")
    )

    for name, profile in export_profiles.items():
        if not isinstance(profile, dict):
            continue
        ready = profile.get("ready") if profile.get("ready") is not None else profile.get("available", False)
        if name == "raw":
            profile["recommended_artifact"] = "raw"
        elif name == "mobile_ar":
            profile["recommended_artifact"] = "none" if not ready else ("normalized" if norm_valid else "raw")
        else:
            profile["recommended_artifact"] = "normalized" if norm_valid else "raw"

    return export_profiles
