"""
GLB export manifest — Sprint 6.

Wraps a BlenderWorkerResult into the `blender_cleanup` manifest block
and (optionally) writes a side-car blender_cleanup.json.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from modules.utils.file_persistence import atomic_write_json

log = logging.getLogger(__name__)


def build_blender_cleanup_block(
    worker_result,  # BlenderWorkerResult
    original_mesh_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert BlenderWorkerResult to manifest block.

    Returned dict is merged into manifest.json under `blender_cleanup`.
    """
    block: Dict[str, Any] = {
        "status": worker_result.status,
        "output_glb": worker_result.output_glb,
        "blender_version": worker_result.blender_version,
        "elapsed_seconds": worker_result.elapsed_seconds,
        "reason": worker_result.reason,
        "original_mesh_path": original_mesh_path,
    }
    return block


def write_blender_cleanup_sidecar(job_dir: "str | Path", block: Dict[str, Any]) -> None:
    """Write blender_cleanup.json next to manifest.json."""
    try:
        atomic_write_json(Path(job_dir) / "blender_cleanup.json", block)
    except Exception as exc:
        log.warning(f"Failed to write blender_cleanup.json: {exc}")
