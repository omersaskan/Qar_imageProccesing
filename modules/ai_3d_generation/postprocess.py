"""
AI 3D generation postprocessing stubs.

Calls existing Blender/gltf-transform tools when available; no-op otherwise.
All functions return structured metadata — never raise.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("ai_3d_generation.postprocess")


def normalize_glb_if_available(glb_path: Optional[str]) -> Dict[str, Any]:
    """Stub: normalize GLB coordinate system. No-op for now."""
    if not glb_path or not Path(glb_path).exists():
        return {"applied": False, "reason": "glb_missing"}
    return {"applied": False, "reason": "not_implemented_yet"}


def optimize_glb_if_available(glb_path: Optional[str]) -> Dict[str, Any]:
    """Stub: run gltf-transform optimize. No-op for now."""
    if not glb_path or not Path(glb_path).exists():
        return {"applied": False, "reason": "glb_missing"}
    try:
        from modules.export_pipeline.gltf_transform_optimizer import GltfTransformOptimizer
        opt = GltfTransformOptimizer()
        if not opt.is_available():
            return {"applied": False, "reason": "gltf_transform_unavailable"}
        result = opt.optimize(glb_path, glb_path)
        return {"applied": True, "result": result}
    except ImportError:
        logger.debug("GLB optimizer not configured (class not found)")
        return {"applied": False, "reason": "optimizer_not_configured"}
    except Exception as exc:
        logger.debug("GLB optimize skipped: %s", exc)
        return {"applied": False, "reason": "optimizer_not_configured"}


def validate_glb_if_available(glb_path: Optional[str]) -> Dict[str, Any]:
    """Stub: run gltf-validator. No-op for now."""
    if not glb_path or not Path(glb_path).exists():
        return {"applied": False, "reason": "glb_missing"}
    try:
        from modules.qa_validation.gltf_validator import GltfValidator
        v = GltfValidator()
        if not v.is_available():
            return {"applied": False, "reason": "gltf_validator_unavailable"}
        result = v.validate(glb_path)
        return {"applied": True, "result": result}
    except ImportError:
        logger.debug("GLB validator not configured (class not found)")
        return {"applied": False, "reason": "validator_not_configured"}
    except Exception as exc:
        logger.debug("GLB validate skipped: %s", exc)
        return {"applied": False, "reason": "validator_not_configured"}


def run_postprocess(
    glb_path: Optional[str],
    enabled: bool = True,
) -> Dict[str, Any]:
    """Run all postprocess steps and return aggregated metadata."""
    if not enabled:
        return {"enabled": False, "normalize": {}, "optimize": {}, "validate": {}}
    return {
        "enabled": True,
        "normalize": normalize_glb_if_available(glb_path),
        "optimize":  optimize_glb_if_available(glb_path),
        "validate":  validate_glb_if_available(glb_path),
    }
