"""
Pipeline decision router for AI 3D generation.

decide_asset_pipeline() is a lightweight classification helper that maps
(input_type, user_intent, capture_quality) → recommended pipeline name.

Does NOT rewrite existing product flows. Returns metadata only.
"""
from __future__ import annotations

from typing import Optional, Dict, Any


# Pipeline name constants
AI_GENERATED_3D    = "ai_generated_3d"
REAL_RECONSTRUCTION = "real_reconstruction"
DEPTH_STUDIO       = "depth_studio"


def decide_asset_pipeline(
    input_type: str,                   # "image" | "video"
    user_intent: str = "default",      # "default" | "advanced" | "debug" | "fast"
    capture_quality: Optional[str] = None,  # "good" | "fair" | "poor" | None
) -> Dict[str, Any]:
    """
    Return a decision dict:
      pipeline         : str  (recommended pipeline name)
      reason           : str
      fallback_pipeline: str | None
      notes            : list[str]

    Rules (in priority order):
      debug intent               → depth_studio
      image input (any intent)   → ai_generated_3d
      video + advanced intent    → real_reconstruction  (if capture_quality good/fair)
      video + good quality       → real_reconstruction
      everything else            → ai_generated_3d  (with real_reconstruction as fallback)
    """
    notes: list[str] = ["ai_generated_not_true_scan"]

    if user_intent == "debug":
        return _decision(DEPTH_STUDIO, "debug_intent_requested",
                         fallback=AI_GENERATED_3D, notes=notes)

    if input_type == "image":
        return _decision(AI_GENERATED_3D, "single_image_best_served_by_ai_generation",
                         fallback=DEPTH_STUDIO, notes=notes)

    # video path
    good_capture = capture_quality in ("good", "fair")
    if user_intent == "advanced" and good_capture:
        return _decision(REAL_RECONSTRUCTION, "advanced_intent_with_good_video_capture",
                         fallback=AI_GENERATED_3D, notes=notes)

    if good_capture and user_intent != "fast":
        notes.append("real_reconstruction_available_as_advanced_option")
        return _decision(AI_GENERATED_3D, "video_default_ai_generation_first",
                         fallback=REAL_RECONSTRUCTION, notes=notes)

    return _decision(AI_GENERATED_3D, "video_fast_or_low_quality_ai_generation",
                     fallback=DEPTH_STUDIO, notes=notes)


def _decision(pipeline: str, reason: str,
              fallback: Optional[str], notes: list) -> Dict[str, Any]:
    return {
        "pipeline": pipeline,
        "reason": reason,
        "fallback_pipeline": fallback,
        "notes": notes,
    }
