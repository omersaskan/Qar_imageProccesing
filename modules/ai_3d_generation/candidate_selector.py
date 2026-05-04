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
    Compute a deterministic score for a single candidate, including preprocessing quality.
    """
    breakdown: Dict[str, Any] = {
        "provider_ok": 0.0,
        "output_exists": 0.0,
        "output_size": 0.0,
        "warning_penalty": 0.0,
        "error_penalty": 0.0,
        "background_removed_bonus": 0.0,
        "rembg_bonus": 0.0,
        "foreground_ratio_score": 0.0,
        "fallback_penalty": 0.0,
        "bbox_sanity_score": 0.0,
    }
    score = 0.0

    # 1. Provider status
    provider_status = meta.get("provider_status", "failed")
    if provider_status == "ok":
        score += _SCORE_PROVIDER_OK
        breakdown["provider_ok"] = _SCORE_PROVIDER_OK

    # 2. GLB file existence & size
    glb_path = meta.get("output_glb_path")
    if glb_path and Path(glb_path).exists():
        score += _SCORE_GLB_EXISTS
        breakdown["output_exists"] = _SCORE_GLB_EXISTS
        try:
            size_kb = Path(glb_path).stat().st_size / 1024
            size_bonus = min(size_kb * _SCORE_GLB_SIZE_PER_KB, _SCORE_GLB_SIZE_MAX)
            score += size_bonus
            breakdown["output_size"] = round(size_bonus, 2)
        except Exception:
            pass

    # 3. Prepared image existence
    prep_path = meta.get("prepared_image_path")
    if prep_path and Path(prep_path).exists():
        score += _SCORE_PREPARED_EXISTS
        # Note: no specific breakdown key for this, usually combined in provider/output

    # 4. Preprocessing Quality
    pre = meta.get("preprocessing", {})
    
    # Background Removal Bonus
    if pre.get("background_removed"):
        score += 8.0
        breakdown["background_removed_bonus"] += 8.0
        
    if pre.get("mask_source") == "rembg":
        score += 5.0
        breakdown["rembg_bonus"] += 5.0
        
    if pre.get("bbox_source") == "rembg_alpha":
        score += 4.0
        breakdown["rembg_bonus"] += 4.0 # Combine rembg-specific metrics

    # Foreground Ratio
    ratio = pre.get("foreground_ratio_estimate")
    if ratio is not None:
        if 0.15 <= ratio <= 0.70:
            score += 6.0
            breakdown["foreground_ratio_score"] += 6.0
        elif 0.05 <= ratio < 0.15 or 0.70 < ratio <= 0.85:
            score += 3.0
            breakdown["foreground_ratio_score"] += 3.0
        elif ratio < 0.03:
            score -= 8.0
            breakdown["foreground_ratio_score"] -= 8.0
        elif ratio > 0.90:
            score -= 6.0
            breakdown["foreground_ratio_score"] -= 6.0

    # Fallback Penalties
    warnings = meta.get("warnings", [])
    if pre.get("crop_method") == "fallback_center_crop":
        score -= 5.0
        breakdown["fallback_penalty"] -= 5.0
    
    if any("rembg_unavailable" in w for w in warnings):
        score -= 4.0
        breakdown["fallback_penalty"] -= 4.0
    elif any("rembg_failed" in w for w in warnings):
        score -= 6.0
        breakdown["fallback_penalty"] -= 6.0
    elif any("rembg_empty_alpha" in w for w in warnings):
        score -= 8.0
        breakdown["fallback_penalty"] -= 8.0

    # BBox Sanity
    bbox = pre.get("bbox")
    crop_bbox = pre.get("crop_bbox")
    cw = pre.get("crop_width", 0)
    ch = pre.get("crop_height", 0)
    
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and \
       isinstance(crop_bbox, (list, tuple)) and len(crop_bbox) == 4:
        score += 3.0
        breakdown["bbox_sanity_score"] += 3.0
    else:
        score -= 5.0
        breakdown["bbox_sanity_score"] -= 5.0

    if cw <= 0 or ch <= 0:
        score -= 5.0
        breakdown["bbox_sanity_score"] -= 5.0
    else:
        # Check for extremely small crop relative to original (proxy for noise/error)
        orig_w = pre.get("original_width", 1)
        orig_h = pre.get("original_height", 1)
        crop_area = cw * ch
        orig_area = orig_w * orig_h
        if crop_area < orig_area * 0.02:
            score -= 4.0
            breakdown["bbox_sanity_score"] -= 4.0

    # General Warning/Error penalties
    if "no_mask_or_bbox_using_center_crop" in warnings:
        # Already penalized via crop_method, but add small additional penalty for manifest consistency
        score += _PENALTY_CENTER_CROP
        breakdown["warning_penalty"] += _PENALTY_CENTER_CROP

    if any(k in [w.split(":")[0] for w in warnings] for k in ("input_video_best_frame_used", "frame_selection_failed")):
        score += _PENALTY_FRAME_SELECTION
        breakdown["warning_penalty"] += _PENALTY_FRAME_SELECTION

    if meta.get("errors"):
        score -= 20.0
        breakdown["error_penalty"] -= 20.0

    # Final clamp and output
    final_score = max(0.0, round(score, 2))
    breakdown["final_score"] = final_score
    return final_score, breakdown


def select_best(
    candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], str]:
    """
    Pick the best successful candidate from a scored list.
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

    def _compact(c, selected):
        pre = c.get("preprocessing", {})
        return {
            "candidate_id": c.get("candidate_id"),
            "score": c.get("score"),
            "status": c.get("status"),
            "provider_status": c.get("provider_status"),
            "selected": selected,
            "background_removed": pre.get("background_removed", False),
            "mask_source": pre.get("mask_source", "none"),
            "foreground_ratio_estimate": pre.get("foreground_ratio_estimate"),
        }

    if not successful:
        compact_ranking = [_compact(c, False) for c in candidates_sorted]
        return None, compact_ranking, "all_candidates_failed"

    # Among successful, pick highest score
    successful.sort(key=lambda c: c.get("score", 0), reverse=True)
    best = successful[0]
    reason = f"highest_score ({best.get('score', 0)})"

    compact_ranking = [
        _compact(c, c.get("candidate_id") == best.get("candidate_id"))
        for c in candidates_sorted
    ]

    return best, compact_ranking, reason
