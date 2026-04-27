"""
SAM2 Prompt Generation Utility
===============================

Generates prompts for SAM2 inference from various sources:
- center_point: frame center or legacy mask centroid
- center_box: legacy mask bounding box if available
- auto: prefers legacy mask bbox if confidence is reasonable,
        otherwise falls back to center point

This module does NOT depend on torch or SAM2.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def generate_prompts(
    frame_shape: Tuple[int, int],
    mode: str = "center_box",
    legacy_mask: Optional[np.ndarray] = None,
    legacy_meta: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = 0.40,
) -> Dict[str, Any]:
    """
    Generate SAM2 prompts for a single frame.

    Args:
        frame_shape: (height, width) of the frame.
        mode: One of "center_point", "center_box", "auto".
        legacy_mask: Optional binary mask from legacy segmentation.
        legacy_meta: Optional metadata dict from legacy segmentation.
        confidence_threshold: Minimum legacy confidence for auto mode
            to trust the legacy bbox.

    Returns:
        dict with keys:
            prompt_mode: str — actual mode used
            prompt_source: str — "legacy_mask", "frame_center", etc.
            bbox: Optional[List[int]] — [x1, y1, x2, y2] if applicable
            points: Optional[List[List[int]]] — [[x, y]] if applicable
            labels: Optional[List[int]] — [1] for foreground points
            confidence: float — confidence in the prompt quality
    """
    h, w = frame_shape

    # Try to extract bbox / centroid from legacy mask
    legacy_bbox = None
    legacy_centroid = None
    legacy_confidence = 0.0

    if legacy_mask is not None and np.any(legacy_mask > 0):
        legacy_bbox, legacy_centroid = _extract_bbox_centroid(legacy_mask)
        if legacy_meta:
            legacy_confidence = float(legacy_meta.get("mask_confidence", 0.0))

    # Resolve "auto" mode
    resolved_mode = mode
    if mode == "auto":
        if legacy_bbox is not None and legacy_confidence >= confidence_threshold:
            resolved_mode = "center_box"
        else:
            resolved_mode = "center_point"
        logger.debug(
            f"Auto prompt resolved to '{resolved_mode}' "
            f"(legacy_conf={legacy_confidence:.2f}, threshold={confidence_threshold})"
        )

    if resolved_mode == "center_box":
        return _make_box_prompt(h, w, legacy_bbox, legacy_centroid, legacy_confidence)
    else:
        return _make_point_prompt(h, w, legacy_centroid, legacy_confidence)


def _extract_bbox_centroid(
    mask: np.ndarray,
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Extract bounding box and centroid from a binary mask."""
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    M = cv2.moments(largest)
    if M["m00"] > 1e-8:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx = x + bw // 2
        cy = y + bh // 2

    return [x, y, x + bw, y + bh], [cx, cy]


def _make_box_prompt(
    h: int,
    w: int,
    legacy_bbox: Optional[List[int]],
    legacy_centroid: Optional[List[int]],
    legacy_confidence: float,
) -> Dict[str, Any]:
    """Create a bounding-box prompt, preferring legacy mask bbox."""
    if legacy_bbox is not None:
        return {
            "prompt_mode": "center_box",
            "prompt_source": "legacy_mask",
            "bbox": legacy_bbox,
            "points": [legacy_centroid] if legacy_centroid else [[w // 2, h // 2]],
            "labels": [1],
            "confidence": min(1.0, legacy_confidence * 1.1),
        }
    # Fallback: use center 50% of frame as box
    pad_x = w // 4
    pad_y = h // 4
    return {
        "prompt_mode": "center_box",
        "prompt_source": "frame_center",
        "bbox": [pad_x, pad_y, w - pad_x, h - pad_y],
        "points": [[w // 2, h // 2]],
        "labels": [1],
        "confidence": 0.3,
    }


def _make_point_prompt(
    h: int,
    w: int,
    legacy_centroid: Optional[List[int]],
    legacy_confidence: float,
) -> Dict[str, Any]:
    """Create a center-point prompt, preferring legacy mask centroid."""
    if legacy_centroid is not None:
        return {
            "prompt_mode": "center_point",
            "prompt_source": "legacy_mask",
            "bbox": None,
            "points": [legacy_centroid],
            "labels": [1],
            "confidence": min(1.0, legacy_confidence * 1.05),
        }
    return {
        "prompt_mode": "center_point",
        "prompt_source": "frame_center",
        "bbox": None,
        "points": [[w // 2, h // 2]],
        "labels": [1],
        "confidence": 0.25,
    }
