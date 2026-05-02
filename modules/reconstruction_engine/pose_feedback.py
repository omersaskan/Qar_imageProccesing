"""
Pose feedback — Sprint 5.

Orchestrates the full Sprint 5 pose-backed pipeline for a completed
reconstruction attempt and returns a `pose_backed_coverage` block
ready for the manifest.

Entry point:
    generate_pose_feedback(attempt_dir, input_frame_count) -> dict

The block is merged into manifest.json by runner._finalize_best_attempt
when POSE_BACKED_COVERAGE_ENABLED=true (default false).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .colmap_sparse_parser import load_sparse_model
from .pose_coverage_matrix import build_coverage_matrix, _unavailable
from .orbit_validation import validate_orbit, OrbitThresholds


def generate_pose_feedback(
    attempt_dir: "str | Path",
    input_frame_count: int = 0,
    thresholds: Optional[OrbitThresholds] = None,
) -> Dict[str, Any]:
    """
    Build the full pose_backed_coverage manifest block.

    Returns a dict with keys:
      status, coverage, orbit_validation, sparse_model_dir
    Never raises — all failures produce status="unavailable".
    """
    attempt_dir = Path(attempt_dir)
    try:
        cameras, images, model_dir = load_sparse_model(attempt_dir)
        if not images:
            return _build_block(
                _unavailable("sparse/cameras.txt or images.txt not found"),
                None,
                input_frame_count,
                thresholds,
            )
        registered_ratio = len(images) / input_frame_count if input_frame_count > 0 else None
        coverage = build_coverage_matrix(images)
        # Annotate registered_ratio directly on coverage for convenience
        if registered_ratio is not None:
            coverage["registered_ratio"] = round(registered_ratio, 4)
        return _build_block(coverage, model_dir, input_frame_count, thresholds)
    except Exception as exc:
        logging.warning(f"pose_feedback: unexpected error: {exc}")
        return _build_block(_unavailable(str(exc)[:200]), None, input_frame_count, thresholds)


def _build_block(
    coverage: Dict[str, Any],
    model_dir: Optional[Path],
    input_frame_count: int,
    thresholds: Optional[OrbitThresholds],
) -> Dict[str, Any]:
    orbit = validate_orbit(
        coverage,
        total_input_frames=input_frame_count,
        thresholds=thresholds,
    )
    return {
        "status": coverage.get("status", "unavailable"),
        "sparse_model_dir": str(model_dir) if model_dir else None,
        "coverage": coverage,
        "orbit_validation": orbit.to_dict(),
    }
