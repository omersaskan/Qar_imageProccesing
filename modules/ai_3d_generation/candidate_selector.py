"""
Phase 1 — Deterministic candidate scoring and selection.

Pure helper module. Does NOT call SF3D or any ML provider.

Responsibilities:
  - Score a candidate result dict using deterministic rules.
  - Select the best successful candidate from a list.
  - Handle all-failed scenario.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("ai_3d_generation.candidate_selector")

# ── Score weights ─────────────────────────────────────────────────────────────

_SCORE_PROVIDER_OK = 50.0        # Base score for provider status == "ok"
_SCORE_GLB_EXISTS  = 20.0        # GLB file exists on disk
_SCORE_GLB_SIZE_PER_KB = 0.01    # Per KB of GLB file size
_SCORE_GLB_SIZE_MAX = 15.0       # Cap on size bonus
_SCORE_PREPARED_EXISTS = 5.0     # ai3d_input.png exists
_PENALTY_CENTER_CROP = -3.0      # Center-crop fallback was used (no mask/bbox)
_PENALTY_FRAME_SELECTION = -2.0  # Frame selection warning present


def score_candidate(meta: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Compute a deterministic score for a single candidate.

    Parameters
    ----------
    meta : dict
        Candidate metadata as returned by candidate_runner.
        Expected keys: provider_status, output_glb_path, prepared_image_path,
        warnings, status.

    Returns
    -------
    (score, breakdown) : (float, dict)
        The numeric score and a dict explaining each component.
    """
    breakdown: Dict[str, Any] = {}
    score = 0.0

    # 1. Provider status
    provider_status = meta.get("provider_status", "failed")
    if provider_status == "ok":
        score += _SCORE_PROVIDER_OK
        breakdown["provider_ok"] = _SCORE_PROVIDER_OK
    else:
        breakdown["provider_ok"] = 0.0

    # 2. GLB file existence & size
    glb_path = meta.get("output_glb_path")
    if glb_path and Path(glb_path).exists():
        score += _SCORE_GLB_EXISTS
        breakdown["glb_exists"] = _SCORE_GLB_EXISTS
        try:
            size_kb = Path(glb_path).stat().st_size / 1024
            size_bonus = min(size_kb * _SCORE_GLB_SIZE_PER_KB, _SCORE_GLB_SIZE_MAX)
            score += size_bonus
            breakdown["glb_size_kb"] = round(size_kb, 1)
            breakdown["glb_size_bonus"] = round(size_bonus, 2)
        except Exception:
            breakdown["glb_size_bonus"] = 0.0
    else:
        breakdown["glb_exists"] = 0.0
        breakdown["glb_size_bonus"] = 0.0

    # 3. Prepared image exists
    prep_path = meta.get("prepared_image_path")
    if prep_path and Path(prep_path).exists():
        score += _SCORE_PREPARED_EXISTS
        breakdown["prepared_exists"] = _SCORE_PREPARED_EXISTS
    else:
        breakdown["prepared_exists"] = 0.0

    # 4. Penalties
    warnings = meta.get("warnings", [])
    if "no_mask_or_bbox_using_center_crop" in warnings:
        score += _PENALTY_CENTER_CROP
        breakdown["center_crop_penalty"] = _PENALTY_CENTER_CROP

    if "input_video_best_frame_used" in warnings or "frame_selection_failed" in [
        w.split(":")[0] for w in warnings
    ]:
        score += _PENALTY_FRAME_SELECTION
        breakdown["frame_selection_penalty"] = _PENALTY_FRAME_SELECTION

    breakdown["total"] = round(score, 2)
    return round(score, 2), breakdown


def select_best(
    candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], str]:
    """
    Pick the best successful candidate from a scored list.

    Parameters
    ----------
    candidates : list[dict]
        Each dict must have at least: candidate_id, status, score, score_breakdown.

    Returns
    -------
    (best, ranking, reason) : (dict | None, list[dict], str)
        - best: the winning candidate metadata, or None if all failed.
        - ranking: all candidates sorted by score descending.
        - reason: human-readable selection reason.
    """
    if not candidates:
        return None, [], "no_candidates"

    # Filter to successful candidates
    successful = []
    for c in candidates:
        has_glb = bool(c.get("output_glb_path") and Path(c.get("output_glb_path")).exists())
        if c.get("provider_status") == "ok" and has_glb:
            successful.append(c)

    # Sort all by score descending for ranking
    candidates_sorted = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)

    if not successful:
        compact_ranking = [
            {
                "candidate_id": c.get("candidate_id"),
                "score": c.get("score"),
                "status": c.get("status"),
                "provider_status": c.get("provider_status"),
                "selected": False,
            }
            for c in candidates_sorted
        ]
        return None, compact_ranking, "all_candidates_failed"

    # Among successful, pick highest score
    successful.sort(key=lambda c: c.get("score", 0), reverse=True)
    best = successful[0]
    reason = f"highest_score ({best.get('score', 0)})"

    compact_ranking = [
        {
            "candidate_id": c.get("candidate_id"),
            "score": c.get("score"),
            "status": c.get("status"),
            "provider_status": c.get("provider_status"),
            "selected": c.get("candidate_id") == best.get("candidate_id"),
        }
        for c in candidates_sorted
    ]

    return best, compact_ranking, reason
