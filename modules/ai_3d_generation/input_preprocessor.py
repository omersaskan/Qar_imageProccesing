"""
Input preprocessor for AI 3D generation.

Prepares a canonical square input image from a source image or video frame:
  - Object-centered crop if mask/bbox is provided
  - Square pad with neutral background
  - Resize to target input_size
  - Save as ai3d_input.png

Does not require SAM2. Uses safe center-crop when segmentation is unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("ai_3d_generation.input_preprocessor")

_DEFAULT_INPUT_SIZE = 512
_NEUTRAL_BG = (127, 127, 127)   # neutral grey for padding


def preprocess_input(
    source_image_path: str,
    output_dir: str,
    input_size: int = _DEFAULT_INPUT_SIZE,
    bbox: Optional[Tuple[int, int, int, int]] = None,   # (x0, y0, x1, y1) tight crop hint
    mask: "Optional[np.ndarray]" = None,                 # uint8 H×W 255=subject
    pad_color: Tuple[int, int, int] = _NEUTRAL_BG,
) -> Dict[str, Any]:
    """
    Prepare canonical SF3D input image.

    Returns:
      input_type         : "image"
      source_image_path  : str
      prepared_image_path: str
      crop_bbox          : list | None
      input_size         : int
      background_removed : bool
      warnings           : list[str]
    """
    import cv2
    import numpy as np

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    background_removed = False
    actual_bbox = None

    img = cv2.imread(source_image_path)
    if img is None:
        return _preprocess_error(source_image_path, output_dir, "Cannot load source image")

    h, w = img.shape[:2]

    # ── 1. Determine crop region ──────────────────────────────────────────────
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)
        actual_bbox = [x0, y0, x1, y1]
    elif mask is not None:
        actual_bbox = _bbox_from_mask(mask, h, w)
        if actual_bbox is None:
            warnings.append("mask_empty_using_full_image")
    else:
        warnings.append("no_mask_or_bbox_using_center_crop")
        # Safe center crop: 80% of shorter dimension
        side = int(min(h, w) * 0.80)
        cx, cy = w // 2, h // 2
        actual_bbox = [cx - side // 2, cy - side // 2,
                       cx + side // 2, cy + side // 2]

    # ── 2. Crop ───────────────────────────────────────────────────────────────
    if actual_bbox:
        x0, y0, x1, y1 = actual_bbox
        # Add 10% padding around the bbox
        pad_x = max(4, int((x1 - x0) * 0.10))
        pad_y = max(4, int((y1 - y0) * 0.10))
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(w, x1 + pad_x)
        y1 = min(h, y1 + pad_y)
        cropped = img[y0:y1, x0:x1]
    else:
        cropped = img

    # ── 3. Square pad ─────────────────────────────────────────────────────────
    squared = _square_pad(cropped, pad_color)

    # ── 4. Resize ─────────────────────────────────────────────────────────────
    resized = cv2.resize(squared, (input_size, input_size), interpolation=cv2.INTER_LANCZOS4)

    # ── 5. Save ───────────────────────────────────────────────────────────────
    out_path = str(out_dir / "ai3d_input.png")
    cv2.imwrite(out_path, resized)

    logger.debug("Preprocessed input → %s  bbox=%s  size=%d", out_path, actual_bbox, input_size)

    return {
        "input_type": "image",
        "source_image_path": source_image_path,
        "prepared_image_path": out_path,
        "crop_bbox": actual_bbox,
        "input_size": input_size,
        "background_removed": background_removed,
        "warnings": warnings,
    }


def _square_pad(img: "np.ndarray", color: Tuple[int, int, int]) -> "np.ndarray":
    import numpy as np
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    out = np.full((side, side, 3), color, dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    out[y_off:y_off + h, x_off:x_off + w] = img
    return out


def _bbox_from_mask(mask: "np.ndarray", h: int, w: int) -> Optional[list]:
    import numpy as np
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _preprocess_error(source: str, output_dir: str, reason: str) -> Dict[str, Any]:
    return {
        "input_type": "image",
        "source_image_path": source,
        "prepared_image_path": None,
        "crop_bbox": None,
        "input_size": 0,
        "background_removed": False,
        "warnings": [f"preprocess_failed:{reason}"],
    }
