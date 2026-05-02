"""Basic image integrity and size checks before depth inference."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple


def check_image(image_path: str, max_pixels: int = 4096 * 4096) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Returns (ok, reason, metadata).
    Avoids hard import failures — cv2 and PIL are both optional.
    """
    meta: Dict[str, Any] = {"path": image_path, "width": 0, "height": 0, "channels": 0}

    if not Path(image_path).exists():
        return False, "File not found", meta

    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return False, "cv2 cannot decode image", meta
        h, w = img.shape[:2]
        c = img.shape[2] if img.ndim == 3 else 1
        meta.update({"width": w, "height": h, "channels": c})
        if w * h > max_pixels:
            return False, f"Image too large ({w}x{h} = {w*h} px > {max_pixels})", meta
        if w < 64 or h < 64:
            return False, f"Image too small ({w}x{h})", meta
        return True, "", meta
    except ImportError:
        pass

    try:
        from PIL import Image
        with Image.open(image_path) as img:
            w, h = img.size
            meta.update({"width": w, "height": h, "channels": len(img.getbands())})
            if w * h > max_pixels:
                return False, f"Image too large ({w}x{h})", meta
            if w < 64 or h < 64:
                return False, f"Image too small ({w}x{h})", meta
            return True, "", meta
    except ImportError:
        return False, "Neither cv2 nor PIL available for image check", meta
    except Exception as e:
        return False, f"Image decode error: {e}", meta
