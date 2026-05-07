"""
Safe normalized copy of a provider GLB.

create_normalized_copy(raw_glb_path, output_dir, normalization_analysis, *, enabled) -> dict

Produces normalized.glb in output_dir using safe centering and ground-alignment
transforms only. Never overwrites raw_glb_path.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_NORMALIZED_COPY_ENABLED = (
    os.environ.get("AI_3D_NORMALIZED_COPY_ENABLED", "true").lower() == "true"
)
_APPLY_TRANSFORM = (
    os.environ.get("AI_3D_APPLY_TRANSFORM_TO_NORMALIZED_COPY", "true").lower() == "true"
)
_MAX_GROUND_OFFSET = 500.0  # skip ground-align if |ymin| exceeds this
_OVERWRITE_RAW = False       # AI_3D_OVERWRITE_RAW_OUTPUT is ALWAYS ignored


def create_normalized_copy(
    raw_glb_path: Optional[str],
    output_dir: Optional[str],
    normalization_analysis: Optional[Dict[str, Any]] = None,
    *,
    enabled: bool = True,
) -> Dict[str, Any]:
    """
    Create a safe normalized copy as normalized.glb in output_dir.

    Allowed transforms (applied to vertex positions, preserving scale and orientation):
    - Translate XZ to center model at origin
    - Translate Y so minimum Y (ground) = 0

    Never modifies the raw GLB. Returns a dict with shape described in AQ2 spec.
    """
    _enabled = enabled and _NORMALIZED_COPY_ENABLED
    result: Dict[str, Any] = {
        "enabled": _enabled,
        "available": False,
        "applied": False,
        "path": None,
        "raw_preserved": True,
        "transform_applied": {
            "centered": False,
            "ground_aligned": False,
            "scaled": False,
            "orientation_changed": False,
        },
        "before": None,
        "after": None,
        "validation": {"valid": None, "issues": [], "warnings": []},
        "warnings": [],
        "error": None,
    }

    if not _enabled:
        return result

    if not raw_glb_path or not Path(raw_glb_path).exists():
        result["warnings"].append("raw_glb_missing")
        return result

    if not output_dir:
        result["warnings"].append("output_dir_missing")
        return result

    out_path = str(Path(output_dir) / "normalized.glb")

    # Hard guard: never overwrite the source file
    if Path(out_path).resolve() == Path(raw_glb_path).resolve():
        result["warnings"].append("normalized_copy_would_overwrite_raw")
        result["error"] = "normalized_copy_would_overwrite_raw"
        return result

    norm_warnings: List[str] = (normalization_analysis or {}).get("warnings") or []

    try:
        import trimesh
        import numpy as np

        scene = trimesh.load(raw_glb_path, force="scene")

        # Collect all mesh vertices (local space — acceptable for simple SF3D scenes)
        all_verts = _collect_vertices(scene)
        if not all_verts:
            shutil.copy2(raw_glb_path, out_path)
            result["available"] = True
            result["applied"] = False
            result["warnings"].append("no_geometry_found_copy_only")
            _validate_normalized(result, out_path)
            return result

        verts = np.concatenate(all_verts, axis=0)
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        center = (mins + maxs) / 2.0

        result["before"] = {
            "bounds": [[round(float(mins[i]), 4) for i in range(3)],
                       [round(float(maxs[i]), 4) for i in range(3)]],
            "dimensions": {
                "x": round(float(maxs[0] - mins[0]), 4),
                "y": round(float(maxs[1] - mins[1]), 4),
                "z": round(float(maxs[2] - mins[2]), 4),
            },
            "center": [round(float(center[i]), 4) for i in range(3)],
            "ground_offset": round(float(mins[1]), 4),
        }

        if not _APPLY_TRANSFORM:
            shutil.copy2(raw_glb_path, out_path)
            result["available"] = True
            result["applied"] = False
            result["warnings"].append("transform_disabled_by_flag")
            _validate_normalized(result, out_path)
            return result

        do_center = "model_not_centered" in norm_warnings
        do_ground = (
            "ground_alignment_uncertain" in norm_warnings
            and abs(float(mins[1])) < _MAX_GROUND_OFFSET
        )

        if not (do_center or do_ground):
            shutil.copy2(raw_glb_path, out_path)
            result["available"] = True
            result["applied"] = False
            result["warnings"].append("normalization_confidence_low")
            _validate_normalized(result, out_path)
            return result

        translate = np.array([
            -float(center[0]) if do_center else 0.0,
            -float(mins[1]) if do_ground else 0.0,
            -float(center[2]) if do_center else 0.0,
        ])

        _apply_translation(scene, translate)
        scene.export(out_path)

        result["applied"] = True
        result["transform_applied"]["centered"] = do_center
        result["transform_applied"]["ground_aligned"] = do_ground

        after_verts = _collect_vertices(scene)
        if after_verts:
            av = np.concatenate(after_verts, axis=0)
            amins, amaxs = av.min(axis=0), av.max(axis=0)
            acenter = (amins + amaxs) / 2.0
            result["after"] = {
                "bounds": [[round(float(amins[i]), 4) for i in range(3)],
                           [round(float(amaxs[i]), 4) for i in range(3)]],
                "dimensions": {
                    "x": round(float(amaxs[0] - amins[0]), 4),
                    "y": round(float(amaxs[1] - amins[1]), 4),
                    "z": round(float(amaxs[2] - amins[2]), 4),
                },
                "center": [round(float(acenter[i]), 4) for i in range(3)],
                "ground_offset": round(float(amins[1]), 4),
            }

        result["available"] = True
        _validate_normalized(result, out_path)

    except ImportError:
        shutil.copy2(raw_glb_path, out_path)
        result["available"] = True
        result["applied"] = False
        result["warnings"].append("trimesh_unavailable_copy_only")
        _validate_normalized(result, out_path)

    except Exception as exc:
        result["error"] = _sanitize(exc)
        result["warnings"].append("normalized_copy_failed")
        log.warning("create_normalized_copy failed: %s", exc)

    return result


def _collect_vertices(scene: Any) -> list:
    try:
        import trimesh
        all_v = []
        if isinstance(scene, trimesh.Scene):
            for mesh in scene.geometry.values():
                if hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
                    all_v.append(mesh.vertices)
        elif hasattr(scene, "vertices") and len(scene.vertices) > 0:
            all_v.append(scene.vertices)
        return all_v
    except Exception:
        return []


def _apply_translation(scene: Any, translate: Any) -> None:
    try:
        import trimesh
        if isinstance(scene, trimesh.Scene):
            for mesh in scene.geometry.values():
                if hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
                    mesh.vertices = mesh.vertices + translate
        elif hasattr(scene, "vertices"):
            scene.vertices = scene.vertices + translate
    except Exception:
        pass


def _validate_normalized(result: Dict[str, Any], out_path: str) -> None:
    if not Path(out_path).exists():
        return
    try:
        from modules.qa_validation.gltf_validator import validate_glb_content
        val = validate_glb_content(out_path)
        result["validation"] = {
            "valid": val.get("valid"),
            "issues": val.get("issues", []),
            "warnings": val.get("warnings", []),
        }
        if not val.get("valid"):
            result["available"] = False
            result["applied"] = False
            result["warnings"].append("normalized_copy_validation_failed")
    except Exception as vexc:
        log.warning("normalized copy validation error: %s", vexc)
        result["warnings"].append("normalized_copy_validation_skipped")


def _sanitize(exc: Exception) -> str:
    msg = str(exc)
    for sep in ("\\", "/"):
        if sep in msg:
            msg = msg.split(sep)[-1]
    return msg[:200]
