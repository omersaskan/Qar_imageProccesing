"""
Quality profile resolver for AI 3D generation.
Handles 'balanced', 'high', and 'ultra' modes with safety clamping.
"""
from typing import Dict, Any, Optional

def resolve_quality_profile(
    quality_mode: str,
    settings: Any,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Resolves a quality profile based on the mode and settings.
    
    Returns a dict with:
        input_size
        texture_resolution
        max_candidates
        video_topk_frames
        bbox_padding_ratio
        foreground_ratio_target
        remesh
        quality_mode (resolved)
        warnings (list)
    """
    profiles = {
        "balanced": {
            "input_size": 768,
            "texture_resolution": 1024,
            "max_candidates": 3,
            "video_topk_frames": 3,
            "bbox_padding_ratio": 0.12,
            "foreground_ratio_target": 0.90,
            "remesh": "none",
        },
        "high": {
            "input_size": 1024,
            "texture_resolution": 1024,
            "max_candidates": 5,
            "video_topk_frames": 5,
            "bbox_padding_ratio": 0.12,
            "foreground_ratio_target": 0.95,
            "remesh": "none",
        },
        "ultra": {
            "input_size": 1024,
            "texture_resolution": 2048,
            "max_candidates": 8,
            "video_topk_frames": 8,
            "bbox_padding_ratio": 0.14,
            "foreground_ratio_target": 0.95,
            "remesh": "none",
        }
    }

    warnings = []
    mode = (quality_mode or "balanced").lower()
    if mode not in profiles:
        warnings.append(f"unknown_quality_mode_fallback_to_balanced:{mode}")
        mode = "balanced"

    resolved = profiles[mode].copy()
    resolved["quality_mode"] = mode

    # Apply overrides if present
    if overrides:
        for key in ["input_size", "texture_resolution", "max_candidates", "video_topk_frames"]:
            if key in overrides and overrides[key] is not None:
                resolved[key] = overrides[key]

    # Safety Clamping
    # 1. input_size max 1024
    if resolved["input_size"] > 1024:
        resolved["input_size"] = 1024
        
    # 2. texture_resolution max 2048
    if resolved["texture_resolution"] > 2048:
        resolved["texture_resolution"] = 2048

    # 3. max_candidates max settings.ai_3d_max_candidates
    limit_max_candidates = getattr(settings, "ai_3d_max_candidates", 5)
    if resolved["max_candidates"] > limit_max_candidates:
        resolved["max_candidates"] = limit_max_candidates

    resolved["warnings"] = warnings
    return resolved
