"""
Subject masking for Depth Studio.

Priority chain:
  1. SAM2 center-box prompt (if available & enabled)
  2. Depth-threshold foreground detection (close = foreground)
  3. Center-crop bbox fallback

Produces:
  mask_overlay.png   — RGB image with coloured mask overlay
  cropped_subject.jpg — tight-cropped subject region
  mask_stats.json     — fg_ratio, bbox, nonzero_pixels, full_frame_fallback_used

Returns a binary uint8 mask (255=subject, 0=background) + stats dict.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger("depth_studio.subject_masker")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_subject_mask(
    image_path: str,
    depth_norm: np.ndarray,          # float32 H×W [0,1]  — 1=far, 0=near
    output_dir: str,
    prompt_box: Optional[Tuple[int, int, int, int]] = None,  # (x0,y0,x1,y1) user hint
    min_fg_ratio: float = 0.02,
    max_fg_ratio: float = 0.80,
) -> Dict[str, Any]:
    """
    Returns:
      mask          : uint8 ndarray H×W (255=subject)
      fg_ratio      : float
      bbox          : [x0,y0,x1,y1] or None
      nonzero_pixels: int
      method_used   : str  ('sam2' | 'depth_threshold' | 'center_crop_fallback')
      full_frame_fallback_used: bool
      mask_path     : str | None
      overlay_path  : str | None
      crop_path     : str | None
      stats_path    : str | None
      warnings      : list[str]
    """
    import cv2
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        return _full_frame_fallback(image_path, depth_norm, output_dir, "image_load_failed")

    h, w = image.shape[:2]
    warnings: list = []

    # ── 1. SAM2 attempt ──────────────────────────────────────────────────────
    mask, method = _try_sam2(image, image_path, prompt_box, h, w)

    # ── 2. Depth-threshold fallback ──────────────────────────────────────────
    if mask is None:
        warnings.append("sam2_unavailable")
        mask, method = _depth_threshold_mask(depth_norm, h, w)

    # ── 3. Validate mask — sanity check ratio ────────────────────────────────
    fg_ratio = float(np.count_nonzero(mask)) / (h * w)
    full_frame_fallback_used = False

    if fg_ratio < min_fg_ratio:
        warnings.append(f"mask_too_sparse_{fg_ratio:.3f}_using_center_crop")
        mask, method = _center_crop_mask(h, w)
        fg_ratio = float(np.count_nonzero(mask)) / (h * w)
        full_frame_fallback_used = True

    if fg_ratio > max_fg_ratio:
        warnings.append(f"mask_too_large_{fg_ratio:.3f}_applying_center_crop")
        # Intersect with center crop to restrict over-expansion
        crop_mask, _ = _center_crop_mask(h, w, fraction=0.6)
        mask = cv2.bitwise_and(mask, crop_mask)
        fg_ratio = float(np.count_nonzero(mask)) / (h * w)
        if fg_ratio < min_fg_ratio:
            mask = crop_mask
            full_frame_fallback_used = True

    # ── 4. Morphological cleanup ─────────────────────────────────────────────
    mask = _clean_mask(mask)
    fg_ratio = float(np.count_nonzero(mask)) / (h * w)

    # ── 5. Compute bbox ──────────────────────────────────────────────────────
    bbox = _mask_bbox(mask)

    # ── 6. Write debug artifacts ─────────────────────────────────────────────
    mask_path    = _write_mask(mask, out_dir)
    overlay_path = _write_overlay(image, mask, out_dir)
    crop_path    = _write_crop(image, bbox, out_dir)
    stats        = {
        "fg_ratio": round(fg_ratio, 4),
        "nonzero_pixels": int(np.count_nonzero(mask)),
        "total_pixels": h * w,
        "bbox": bbox,
        "method_used": method,
        "full_frame_fallback_used": full_frame_fallback_used,
        "warnings": warnings,
    }
    stats_path = str(out_dir / "mask_stats.json")
    Path(stats_path).write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info(
        "Subject mask: method=%s fg_ratio=%.3f bbox=%s fallback=%s",
        method, fg_ratio, bbox, full_frame_fallback_used,
    )

    return {
        "mask": mask,
        "fg_ratio": fg_ratio,
        "bbox": bbox,
        "nonzero_pixels": int(np.count_nonzero(mask)),
        "method_used": method,
        "full_frame_fallback_used": full_frame_fallback_used,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "crop_path": crop_path,
        "stats_path": stats_path,
        "warnings": warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy implementations
# ─────────────────────────────────────────────────────────────────────────────

def _try_sam2(
    image_bgr: np.ndarray,
    image_path: str,
    prompt_box: Optional[Tuple[int, int, int, int]],
    h: int,
    w: int,
) -> Tuple[Optional[np.ndarray], str]:
    """Attempt SAM2 center-box or user-box segmentation."""
    try:
        from modules.operations.settings import settings
        if not settings.sam2_enabled:
            return None, "sam2_disabled"

        from modules.ai_segmentation.sam2_wrapper import probe_sam2_availability
        avail, _, reason = probe_sam2_availability()
        if not avail:
            logger.debug("SAM2 unavailable: %s", reason)
            return None, "sam2_unavailable"

        from modules.ai_segmentation.sam2_wrapper import Sam2ImageSegmenter
        segmenter = Sam2ImageSegmenter()

        # Build prompt box: user-supplied or auto-center-box
        if prompt_box:
            box = list(prompt_box)
        else:
            box = _auto_center_box(h, w, fraction=0.50)

        result = segmenter.segment_frame(image_path, box_prompt=box)
        if result is None or result.get("mask") is None:
            return None, "sam2_no_mask_returned"

        raw_mask = result["mask"]
        if isinstance(raw_mask, np.ndarray):
            # Ensure uint8 binary
            if raw_mask.dtype != np.uint8:
                raw_mask = (raw_mask > 0).astype(np.uint8) * 255
            if raw_mask.shape[:2] != (h, w):
                import cv2
                raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            return raw_mask, "sam2"
    except Exception as e:
        logger.debug("SAM2 attempt failed: %s", e)

    return None, "sam2_failed"


def _depth_threshold_mask(
    depth_norm: np.ndarray,
    h: int,
    w: int,
    near_percentile: float = 30.0,
) -> Tuple[np.ndarray, str]:
    """
    Foreground = pixels closer than near_percentile of depth distribution.
    depth_norm: float32 [0,1] where 0=near, 1=far  (depth_anything convention).
    Invert if convention is reversed (1=near): checked via median heuristic.
    """
    import cv2

    d = depth_norm.copy()
    if d.ndim == 3:
        d = d[:, :, 0]
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convention check: depth_anything returns relative depth where
    # larger value = farther away.  If median > 0.5 the values look inverted.
    # We want low value = near = foreground.
    if np.median(d) > 0.5:
        d = 1.0 - d   # invert so 0=near

    threshold = np.percentile(d, near_percentile)
    binary = (d < threshold).astype(np.uint8) * 255

    # Remove tiny isolated regions
    binary = _clean_mask(binary, open_iter=2, close_iter=4)
    return binary, "depth_threshold"


def _center_crop_mask(
    h: int,
    w: int,
    fraction: float = 0.55,
) -> Tuple[np.ndarray, str]:
    """Elliptical center-crop mask as last-resort fallback."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    ry, rx = int(h * fraction / 2), int(w * fraction / 2)
    import cv2
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return mask, "center_crop_fallback"


def _full_frame_fallback(
    image_path: str,
    depth_norm: np.ndarray,
    output_dir: str,
    reason: str,
) -> Dict[str, Any]:
    h, w = depth_norm.shape[:2] if depth_norm.ndim >= 2 else (256, 256)
    mask = np.full((h, w), 255, dtype=np.uint8)
    return {
        "mask": mask,
        "fg_ratio": 1.0,
        "bbox": [0, 0, w, h],
        "nonzero_pixels": h * w,
        "method_used": "full_frame_fallback",
        "full_frame_fallback_used": True,
        "mask_path": None,
        "overlay_path": None,
        "crop_path": None,
        "stats_path": None,
        "warnings": [f"full_frame_fallback:{reason}"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _auto_center_box(h: int, w: int, fraction: float = 0.50) -> list:
    """Return [x0,y0,x1,y1] center box covering `fraction` of each dimension."""
    pad_x = int(w * (1 - fraction) / 2)
    pad_y = int(h * (1 - fraction) / 2)
    return [pad_x, pad_y, w - pad_x, h - pad_y]


def _clean_mask(
    mask: np.ndarray,
    open_iter: int = 1,
    close_iter: int = 6,
) -> np.ndarray:
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = mask.copy()
    if open_iter:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=open_iter)
    if close_iter:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=close_iter)
    return m


def _mask_bbox(mask: np.ndarray) -> Optional[list]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _write_mask(mask: np.ndarray, out_dir: Path) -> str:
    import cv2
    path = str(out_dir / "subject_mask.png")
    cv2.imwrite(path, mask)
    return path


def _write_overlay(image: np.ndarray, mask: np.ndarray, out_dir: Path) -> str:
    """Green overlay on subject region."""
    import cv2
    overlay = image.copy()
    green = np.zeros_like(image)
    green[:, :] = (0, 200, 80)
    alpha_mask = (mask > 0).astype(np.float32)[:, :, None]
    overlay = (overlay * (1 - 0.45 * alpha_mask) + green * 0.45 * alpha_mask).astype(np.uint8)
    # Draw bbox
    bbox = _mask_bbox(mask)
    if bbox:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 128), 2)
    path = str(out_dir / "mask_overlay.png")
    cv2.imwrite(path, overlay)
    return path


def _write_crop(image: np.ndarray, bbox: Optional[list], out_dir: Path) -> Optional[str]:
    if bbox is None:
        return None
    import cv2
    x0, y0, x1, y1 = bbox
    pad = 16
    h, w = image.shape[:2]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    crop = image[y0:y1, x0:x1]
    path = str(out_dir / "cropped_subject.jpg")
    cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def apply_mask_to_depth(
    depth_norm: np.ndarray,
    mask: np.ndarray,
    bg_fill: float = 1.0,
) -> np.ndarray:
    """
    Zero-out background in depth map (set to bg_fill = far value).
    mask: uint8 (255=subject, 0=background)
    Returns float32 depth with background pushed to bg_fill.
    """
    import cv2
    d = depth_norm.copy()
    m = mask.astype(np.float32) / 255.0
    if m.shape != d.shape[:2]:
        m = cv2.resize(m, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_LINEAR)
    if d.ndim == 3:
        m = m[:, :, None]
    return d * m + bg_fill * (1.0 - m)
