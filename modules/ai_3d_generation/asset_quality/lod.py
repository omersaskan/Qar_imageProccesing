"""
LOD (Level of Detail) planning for AI-generated GLB meshes.

build_lod_plan(mesh_stats, ar_readiness, asset_quality_context) -> dict

LOD generation is disabled by default (plan_only).
Set AI_3D_LOD_GENERATION_ENABLED=true to enable conservative decimation.
Never overwrites the raw GLB.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_LOD_GENERATION_ENABLED = (
    os.environ.get("AI_3D_LOD_GENERATION_ENABLED", "false").lower() == "true"
)

_DEFAULT_TIERS = [
    {"name": "preview",  "target_faces": 10_000,  "recommended": True},
    {"name": "mobile",   "target_faces": 25_000,  "recommended": True},
    {"name": "desktop",  "target_faces": 75_000,  "recommended": False},
]


def build_lod_plan(
    mesh_stats: Dict[str, Any],
    ar_readiness: Dict[str, Any],
    asset_quality_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a LOD plan from mesh stats and AR readiness context.

    Parameters
    ----------
    mesh_stats : dict
        Output of extract_mesh_stats().
    ar_readiness : dict
        Output of assess_ar_readiness().
    asset_quality_context : dict | None
        May contain "glb_path" and "output_dir" for LOD generation.

    Returns
    -------
    dict with keys: enabled, available, generated, strategy, tiers, warnings, recommendations
    """
    warnings: List[str] = []
    recommendations: List[str] = []
    face_count: Optional[int] = (mesh_stats or {}).get("face_count")

    tiers: List[Dict[str, Any]] = []
    for tier_def in _DEFAULT_TIERS:
        tier: Dict[str, Any] = {
            "name": tier_def["name"],
            "target_faces": tier_def["target_faces"],
            "recommended": tier_def["recommended"],
            "path": None,
        }
        if face_count is not None and face_count > tier_def["target_faces"]:
            tier["recommended"] = True
        tiers.append(tier)

    strategy = "plan_only"
    generated = False

    if _LOD_GENERATION_ENABLED:
        strategy = "conservative_decimation"
        ctx = asset_quality_context or {}
        glb_path = ctx.get("glb_path")
        output_dir = ctx.get("output_dir")

        if glb_path and output_dir and Path(glb_path).exists():
            result_tiers = _attempt_lod_generation(
                glb_path, output_dir, tiers, warnings
            )
            if result_tiers is not None:
                tiers = result_tiers
                generated = any(t.get("path") for t in tiers)
        else:
            warnings.append("lod_generation_skipped:missing_paths")

    # Recommendations based on face count
    if face_count is not None:
        if face_count <= _DEFAULT_TIERS[0]["target_faces"]:
            recommendations.append(
                "Mesh already at preview density. LOD may not be necessary."
            )
        elif face_count <= _DEFAULT_TIERS[1]["target_faces"]:
            recommendations.append(
                "Mesh within mobile-ready range. Preview LOD recommended."
            )
        else:
            recommendations.append(
                "Mesh exceeds mobile target. Preview and mobile LODs recommended."
            )

    return {
        "enabled": True,
        "available": True,
        "generated": generated,
        "strategy": strategy,
        "tiers": tiers,
        "warnings": warnings,
        "recommendations": recommendations,
    }


def _attempt_lod_generation(
    glb_path: str,
    output_dir: str,
    tiers: List[Dict[str, Any]],
    warnings: List[str],
) -> Optional[List[Dict[str, Any]]]:
    """
    Attempt conservative decimation via trimesh.
    Writes output_<name>.glb per tier. Validates each result.
    Does NOT overwrite the raw GLB.
    Returns updated tiers list or None on early failure.
    """
    try:
        import trimesh

        scene = trimesh.load(glb_path, force="scene")
        result_tiers: List[Dict[str, Any]] = []

        for tier in tiers:
            if not tier.get("recommended"):
                result_tiers.append({**tier})
                continue

            tier_name = tier["name"]
            target = tier["target_faces"]
            out_path = str(Path(output_dir) / f"output_{tier_name}.glb")

            # Guard: never overwrite source
            if Path(out_path).resolve() == Path(glb_path).resolve():
                warnings.append(f"lod_generation_skipped:{tier_name}:would_overwrite_source")
                result_tiers.append({**tier, "path": None})
                continue

            try:
                decimated = _decimate_scene(scene, target)
                if decimated is None:
                    warnings.append(f"lod_generation_skipped:{tier_name}:decimation_failed")
                    result_tiers.append({**tier, "path": None})
                    continue

                decimated.export(out_path)

                from modules.qa_validation.gltf_validator import validate_glb_content
                val = validate_glb_content(out_path)
                if not val.get("valid"):
                    try:
                        Path(out_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                    warnings.append(f"lod_validation_failed:{tier_name}")
                    result_tiers.append({**tier, "path": None})
                else:
                    result_tiers.append({**tier, "path": out_path})

            except Exception as exc:
                log.warning("LOD generation failed for tier %s: %s", tier_name, exc)
                warnings.append(f"lod_generation_failed:{tier_name}")
                result_tiers.append({**tier, "path": None})

        return result_tiers

    except ImportError:
        warnings.append("lod_generation_skipped:trimesh_unavailable")
        return None
    except Exception as exc:
        log.warning("LOD generation error: %s", exc)
        warnings.append("lod_generation_failed:unexpected_error")
        return None


def _decimate_scene(scene: Any, target_faces: int) -> Optional[Any]:
    """
    Proportionally decimate all meshes in scene to approach target_faces.
    Returns a new Scene or None on failure.
    """
    try:
        import trimesh

        total_faces = sum(
            len(m.faces)
            for m in scene.geometry.values()
            if isinstance(m, trimesh.Trimesh)
        )
        if total_faces <= target_faces:
            return scene

        ratio = target_faces / max(total_faces, 1)
        new_geometries: Dict[str, Any] = {}

        for name, geom in scene.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                new_geometries[name] = geom
                continue
            local_target = max(4, int(len(geom.faces) * ratio))
            try:
                decimated = geom.simplify_quadric_decimation(local_target)
                new_geometries[name] = decimated
            except Exception:
                new_geometries[name] = geom

        try:
            return trimesh.Scene(geometry=new_geometries, graph=scene.graph)
        except TypeError:
            # older trimesh versions that do not accept graph kwarg
            return trimesh.Scene(geometry=new_geometries)

    except Exception as exc:
        log.warning("_decimate_scene failed: %s", exc)
        return None
