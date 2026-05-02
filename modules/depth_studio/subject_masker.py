"""
Subject masking for Depth Studio.

Priority chain:
  1. SAM2 center-box prompt (if available & enabled)
  2. Depth-threshold foreground detection + connected component filtering
  3. Center-crop bbox fallback

Connected component filtering (depth_threshold path):
  - Border-touching large components are classified as background (table/floor).
  - Remaining components are scored and the best subject component is kept.
  - Border-touch criteria: bottom border, left+right both, bbox width > 85%, area > 25%.

Component scoring (higher = better subject):
  - Center proximity (Gaussian)
  - Reasonable area ratio (penalise extremes)
  - Reasonable bbox aspect ratio
  - Depth saliency (mean depth difference from background)
  - prompt_box overlap bonus
  - Border contact penalty

Mask quality gate:
  Returns `mask_quality` field: "ok" | "review" | "low_confidence".

Produces in output_dir:
  subject_mask.png       — binary mask
  mask_overlay.png       — RGB image with green overlay + bbox
  cropped_subject.jpg    — tight-cropped subject component (not full bbox)
  mask_stats.json        — full stats including component info

Returns a binary uint8 mask (255=subject, 0=background) + stats dict.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

logger = logging.getLogger("depth_studio.subject_masker")

# ── Border-touch rejection thresholds ────────────────────────────────────────
_BORDER_WIDTH_RATIO  = 0.85   # bbox spans > 85% image width → likely table plane
_BORDER_AREA_RATIO   = 0.25   # component area > 25% total → large background region
_BOTTOM_MARGIN       = 4      # px within this of bottom border counts as "touches bottom"
_SIDE_MARGIN         = 4      # px within this of left/right counts as "touches side"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_subject_mask(
    image_path: str,
    depth_norm: np.ndarray,          # float32 H×W [0,1]  — 0=near, 1=far
    output_dir: str,
    prompt_box: Optional[Tuple[int, int, int, int]] = None,  # (x0,y0,x1,y1) user hint
    min_fg_ratio: float = 0.02,
    max_fg_ratio: float = 0.80,
) -> Dict[str, Any]:
    """
    Returns dict with:
      mask                      : uint8 ndarray H×W (255=subject)
      fg_ratio                  : float
      bbox                      : [x0,y0,x1,y1] or None
      nonzero_pixels            : int
      method_used               : str
      full_frame_fallback_used  : bool
      mask_quality              : str  ('ok' | 'review' | 'low_confidence')
      component_count           : int
      selected_component_area_ratio : float
      rejected_components       : list[dict]
      mask_path / overlay_path / crop_path / stats_path : str | None
      warnings                  : list[str]
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

    # ── 2. Depth-threshold + component filtering ─────────────────────────────
    component_count = 0
    selected_component_area_ratio = 0.0
    rejected_components: List[dict] = []

    if mask is None:
        warnings.append("sam2_unavailable")
        raw_mask, method = _depth_threshold_mask(depth_norm, h, w)
        mask, comp_stats = _filter_components(
            raw_mask, h, w, depth_norm, prompt_box
        )
        component_count = comp_stats["component_count"]
        selected_component_area_ratio = comp_stats["selected_area_ratio"]
        rejected_components = comp_stats["rejected"]

        if mask is None:
            warnings.append("all_components_rejected_using_center_crop")
            mask, method = _center_crop_mask(h, w)
    else:
        # SAM2 succeeded — still count components for stats
        component_count = _count_components(mask)
        selected_component_area_ratio = float(np.count_nonzero(mask)) / (h * w)

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

    # ── 6. Mask quality assessment ───────────────────────────────────────────
    mask_quality = _assess_quality(
        fg_ratio, bbox, h, w,
        method, full_frame_fallback_used,
        "sam2_unavailable" in warnings,
    )
    if mask_quality != "ok":
        warnings.append(f"mask_quality_{mask_quality}")

    # ── 7. Write debug artifacts ─────────────────────────────────────────────
    mask_path    = _write_mask(mask, out_dir)
    overlay_path = _write_overlay(image, mask, out_dir)
    crop_path    = _write_crop(image, bbox, out_dir)
    stats = {
        "fg_ratio": round(fg_ratio, 4),
        "nonzero_pixels": int(np.count_nonzero(mask)),
        "total_pixels": h * w,
        "bbox": bbox,
        "method_used": method,
        "full_frame_fallback_used": full_frame_fallback_used,
        "mask_quality": mask_quality,
        "component_count": component_count,
        "selected_component_area_ratio": round(selected_component_area_ratio, 4),
        "rejected_components": rejected_components,
        "warnings": warnings,
    }
    stats_path = str(out_dir / "mask_stats.json")
    Path(stats_path).write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info(
        "Subject mask: method=%s fg_ratio=%.3f quality=%s bbox=%s fallback=%s components=%d",
        method, fg_ratio, mask_quality, bbox, full_frame_fallback_used, component_count,
    )

    return {
        "mask": mask,
        "fg_ratio": fg_ratio,
        "bbox": bbox,
        "nonzero_pixels": int(np.count_nonzero(mask)),
        "method_used": method,
        "full_frame_fallback_used": full_frame_fallback_used,
        "mask_quality": mask_quality,
        "component_count": component_count,
        "selected_component_area_ratio": selected_component_area_ratio,
        "rejected_components": rejected_components,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "crop_path": crop_path,
        "stats_path": stats_path,
        "warnings": warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Connected component filtering
# ─────────────────────────────────────────────────────────────────────────────

def _filter_components(
    mask: np.ndarray,
    h: int,
    w: int,
    depth_norm: np.ndarray,
    prompt_box: Optional[Tuple[int, int, int, int]],
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Analyse connected components of `mask`, reject background-like regions,
    score and select the best subject component.

    Returns (selected_mask | None, stats_dict).
    """
    import cv2

    num_labels, labels, stats_arr, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    # stats_arr columns: x, y, w, h, area  (label 0 = background)

    total_pixels = h * w
    accepted = []
    rejected = []

    for label in range(1, num_labels):
        x0 = int(stats_arr[label, cv2.CC_STAT_LEFT])
        y0 = int(stats_arr[label, cv2.CC_STAT_TOP])
        bw = int(stats_arr[label, cv2.CC_STAT_WIDTH])
        bh = int(stats_arr[label, cv2.CC_STAT_HEIGHT])
        area = int(stats_arr[label, cv2.CC_STAT_AREA])
        x1, y1 = x0 + bw, y0 + bh
        area_ratio = area / total_pixels
        cx, cy = float(centroids[label][0]), float(centroids[label][1])

        reject_reason = _background_reject_reason(
            x0, y0, x1, y1, area_ratio, h, w
        )
        if reject_reason:
            rejected.append({
                "label": label,
                "area_ratio": round(area_ratio, 4),
                "bbox": [x0, y0, x1, y1],
                "reject_reason": reject_reason,
            })
            continue

        score = _score_component(
            x0, y0, x1, y1, area_ratio, cx, cy, h, w, depth_norm, prompt_box
        )
        accepted.append({
            "label": label,
            "area_ratio": area_ratio,
            "bbox": [x0, y0, x1, y1],
            "score": score,
        })

    stats = {
        "component_count": num_labels - 1,
        "selected_area_ratio": 0.0,
        "rejected": rejected,
    }

    if not accepted:
        return None, stats

    # Keep the highest-scored component
    best = max(accepted, key=lambda c: c["score"])
    selected_mask = ((labels == best["label"]).astype(np.uint8) * 255)
    stats["selected_area_ratio"] = best["area_ratio"]
    return selected_mask, stats


def _background_reject_reason(
    x0: int, y0: int, x1: int, y1: int,
    area_ratio: float,
    h: int, w: int,
) -> Optional[str]:
    """Return a rejection reason string if this component looks like background, else None."""
    touches_bottom = (y1 >= h - _BOTTOM_MARGIN)
    touches_left   = (x0 <= _SIDE_MARGIN)
    touches_right  = (x1 >= w - _SIDE_MARGIN)
    bbox_width_ratio = (x1 - x0) / w

    # Table / floor plane: bottom border + large area
    if touches_bottom and area_ratio > _BORDER_AREA_RATIO:
        return "bottom_border_large_area"

    # Spans full width + touches bottom: textured table surface
    if touches_bottom and bbox_width_ratio > _BORDER_WIDTH_RATIO:
        return "bottom_border_full_width"

    # Spans full width + very large: floor/background gradient
    if touches_left and touches_right and bbox_width_ratio > _BORDER_WIDTH_RATIO:
        return "full_width_background"

    return None


def _score_component(
    x0: int, y0: int, x1: int, y1: int,
    area_ratio: float,
    cx: float, cy: float,
    h: int, w: int,
    depth_norm: np.ndarray,
    prompt_box: Optional[Tuple[int, int, int, int]],
) -> float:
    """Higher score = more likely subject. Combines several weak signals."""
    score = 0.0

    # Center proximity (Gaussian, sigma = 30% of image half-diagonal)
    ic_x, ic_y = w / 2.0, h / 2.0
    dist = np.sqrt((cx - ic_x) ** 2 + (cy - ic_y) ** 2)
    sigma = 0.30 * np.sqrt(ic_x ** 2 + ic_y ** 2)
    score += 3.0 * float(np.exp(-0.5 * (dist / sigma) ** 2))

    # Reasonable area: penalise very small (<1%) and very large (>40%)
    if 0.01 <= area_ratio <= 0.40:
        score += 2.0
    elif area_ratio < 0.01:
        score -= 1.0

    # Bbox aspect ratio — avoid extreme thin/wide bands (table-edge artefacts)
    bw, bh = x1 - x0, y1 - y0
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    if aspect < 5.0:
        score += 1.0
    else:
        score -= 1.0

    # Depth saliency: near pixels are closer to camera → more likely subject
    if depth_norm is not None and depth_norm.size > 0:
        try:
            import cv2
            d = depth_norm
            if d.ndim == 3:
                d = d[:, :, 0]
            d_resized = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR) if d.shape[:2] != (h, w) else d
            region = d_resized[y0:y1, x0:x1]
            mean_depth = float(region.mean()) if region.size > 0 else 0.5
            # Lower depth = closer = more foreground
            score += (1.0 - mean_depth) * 2.0
        except Exception:
            pass

    # prompt_box overlap bonus
    if prompt_box is not None:
        px0, py0, px1, py1 = prompt_box
        ox0, oy0 = max(x0, px0), max(y0, py0)
        ox1, oy1 = min(x1, px1), min(y1, py1)
        if ox1 > ox0 and oy1 > oy0:
            overlap = (ox1 - ox0) * (oy1 - oy0)
            comp_area = max((x1 - x0) * (y1 - y0), 1)
            score += 4.0 * (overlap / comp_area)

    # Border contact penalty (not a hard reject here — soft penalty)
    if y1 >= h - _BOTTOM_MARGIN:
        score -= 1.5
    if x0 <= _SIDE_MARGIN:
        score -= 0.5
    if x1 >= w - _SIDE_MARGIN:
        score -= 0.5

    return score


def _count_components(mask: np.ndarray) -> int:
    import cv2
    n, _ = cv2.connectedComponents(mask, connectivity=8)
    return max(0, n - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Mask quality gate
# ─────────────────────────────────────────────────────────────────────────────

def _assess_quality(
    fg_ratio: float,
    bbox: Optional[list],
    h: int,
    w: int,
    method: str,
    full_frame_fallback: bool,
    sam2_unavailable: bool,
) -> str:
    """
    Returns 'ok', 'review', or 'low_confidence'.
    These map to the pipeline's final_status partial/ok distinction.
    """
    if full_frame_fallback:
        return "low_confidence"

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        bbox_area_ratio = ((x1 - x0) * (y1 - y0)) / (h * w)
        touches_bottom = (y1 >= h - _BOTTOM_MARGIN)
        touches_left   = (x0 <= _SIDE_MARGIN)
        touches_right  = (x1 >= w - _SIDE_MARGIN)

        if bbox_area_ratio > 0.60:
            return "low_confidence"
        if touches_bottom and fg_ratio > 0.20:
            return "review"
        if touches_left and touches_right:
            return "review"

    if fg_ratio > 0.25 and method == "depth_threshold":
        return "review"

    if method == "depth_threshold" and sam2_unavailable:
        return "review"

    return "ok"


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

        box = list(prompt_box) if prompt_box else _auto_center_box(h, w, fraction=0.50)

        result = segmenter.segment_frame(image_path, box_prompt=box)
        if result is None or result.get("mask") is None:
            return None, "sam2_no_mask_returned"

        raw_mask = result["mask"]
        if isinstance(raw_mask, np.ndarray):
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
    depth_norm: float32 [0,1] where 0=near, 1=far (Depth Anything convention).
    Auto-inverts if median > 0.5 (inverted convention detection).
    """
    import cv2

    d = depth_norm.copy()
    if d.ndim == 3:
        d = d[:, :, 0]
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)

    if np.median(d) > 0.5:
        d = 1.0 - d   # invert so 0=near

    threshold = np.percentile(d, near_percentile)
    binary = (d < threshold).astype(np.uint8) * 255
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
        "mask_quality": "low_confidence",
        "component_count": 0,
        "selected_component_area_ratio": 1.0,
        "rejected_components": [],
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
    import cv2
    overlay = image.copy()
    green = np.zeros_like(image)
    green[:, :] = (0, 200, 80)
    alpha_mask = (mask > 0).astype(np.float32)[:, :, None]
    overlay = (overlay * (1 - 0.45 * alpha_mask) + green * 0.45 * alpha_mask).astype(np.uint8)
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


# ─────────────────────────────────────────────────────────────────────────────
# Depth application helper
# ─────────────────────────────────────────────────────────────────────────────

def apply_mask_to_depth(
    depth_norm: np.ndarray,
    mask: np.ndarray,
    bg_fill: float = 1.0,
) -> np.ndarray:
    """
    Set background pixels to bg_fill (far value) in depth map.
    mask: uint8 (255=subject, 0=background).
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
