"""
Export package builder.

create_export_package(session_dir, raw_glb_path, normalized_copy,
                      cleanup_report, asset_quality, export_profiles) -> dict

Creates export_package/ folder with all audit artifacts.
Returns relative paths only — never exposes absolute filesystem paths.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

_EXPORT_PACKAGE_ENABLED = (
    os.environ.get("AI_3D_EXPORT_PACKAGE_ENABLED", "true").lower() == "true"
)


def create_export_package(
    session_dir: Optional[str],
    raw_glb_path: Optional[str],
    normalized_copy: Dict[str, Any],
    cleanup_report: Dict[str, Any],
    asset_quality: Dict[str, Any],
    export_profiles: Dict[str, Any],
) -> Dict[str, Any]:
    _files: Dict[str, Any] = {
        "raw_glb":             {"available": False, "path": None},
        "normalized_glb":      {"available": False, "path": None},
        "cleanup_report_json": {"available": False, "path": None},
        "cleanup_report_md":   {"available": False, "path": None},
        "asset_quality_json":  {"available": False, "path": None},
        "export_manifest_json":{"available": False, "path": None},
    }
    result: Dict[str, Any] = {
        "enabled": _EXPORT_PACKAGE_ENABLED,
        "available": False,
        "package_dir": None,
        "files": _files,
        "warnings": [],
        "error": None,
    }

    if not _EXPORT_PACKAGE_ENABLED:
        return result

    if not session_dir:
        result["warnings"].append("session_dir_missing")
        return result

    try:
        pkg_dir = Path(session_dir) / "export_package"
        pkg_dir.mkdir(parents=True, exist_ok=True)

        # 1. Raw GLB — reference metadata only (no copy of potentially large file)
        raw_exists = bool(raw_glb_path and Path(raw_glb_path).exists())
        raw_size = None
        if raw_exists:
            try:
                raw_size = Path(raw_glb_path).stat().st_size
            except Exception:
                pass
        ref = {"available": raw_exists, "reference_type": "server_local", "size_bytes": raw_size}
        (pkg_dir / "raw_glb_reference.json").write_text(
            json.dumps(ref, indent=2), encoding="utf-8"
        )
        _files["raw_glb"] = {"available": raw_exists, "path": "export_package/raw_glb_reference.json"}

        # 2. Normalized GLB — copy if available and validated
        norm_valid = (
            normalized_copy.get("available")
            and (normalized_copy.get("validation") or {}).get("valid")
        )
        if norm_valid:
            norm_src = Path(session_dir) / "normalized.glb"
            if norm_src.exists():
                shutil.copy2(str(norm_src), str(pkg_dir / "normalized.glb"))
                _files["normalized_glb"] = {"available": True, "path": "export_package/normalized.glb"}

        # 3 & 4. Cleanup report files — copy from session_dir if written
        for fname, key in [("cleanup_report.json", "cleanup_report_json"),
                           ("cleanup_report.md",   "cleanup_report_md")]:
            src = Path(session_dir) / fname
            if src.exists():
                shutil.copy2(str(src), str(pkg_dir / fname))
                _files[key] = {"available": True, "path": f"export_package/{fname}"}

        # 5. Asset quality JSON
        (pkg_dir / "asset_quality.json").write_text(
            json.dumps(asset_quality, indent=2, default=str), encoding="utf-8"
        )
        _files["asset_quality_json"] = {"available": True, "path": "export_package/asset_quality.json"}

        # 6. Export manifest JSON
        export_manifest = {
            "package_type": "aq2_export_package",
            "raw_glb_available": raw_exists,
            "normalized_glb_available": _files["normalized_glb"]["available"],
            "cleanup_report_available": _files["cleanup_report_json"]["available"],
            "asset_quality_verdict": asset_quality.get("verdict"),
            "asset_quality_score": asset_quality.get("score"),
            "mobile_ar_ready": (export_profiles.get("mobile_ar") or {}).get("ready"),
            "web_preview_ready": (export_profiles.get("web_preview") or {}).get("ready"),
            "desktop_high_ready": (export_profiles.get("desktop_high") or {}).get("ready"),
        }
        (pkg_dir / "export_manifest.json").write_text(
            json.dumps(export_manifest, indent=2), encoding="utf-8"
        )
        _files["export_manifest_json"] = {"available": True, "path": "export_package/export_manifest.json"}

        result.update({"available": True, "package_dir": "export_package"})

    except Exception as exc:
        msg = str(exc)
        for sep in ("\\", "/"):
            if sep in msg:
                msg = msg.split(sep)[-1]
        result["error"] = msg[:200]
        result["warnings"].append("export_package_failed")
        log.warning("create_export_package failed: %s", exc)

    return result
