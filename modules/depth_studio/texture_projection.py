"""Project original RGB image onto mesh as texture."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


def prepare_texture(
    source_image_path: str,
    output_dir: str,
    max_size: int = 2048,
) -> Dict[str, Any]:
    """
    Copy/resize source image for use as mesh texture.
    Returns dict with texture_path and dimensions.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / "texture.jpg")

    try:
        import cv2
        img = cv2.imread(source_image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {source_image_path}")
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        h, w = img.shape[:2]
        return {"status": "ok", "texture_path": out_path, "width": w, "height": h}
    except ImportError:
        pass

    try:
        from PIL import Image
        img = Image.open(source_image_path).convert("RGB")
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
        img.save(out_path, "JPEG", quality=92)
        return {"status": "ok", "texture_path": out_path, "width": img.width, "height": img.height}
    except ImportError:
        return {"status": "unavailable", "reason": "Neither cv2 nor PIL available", "texture_path": None}
    except Exception as e:
        return {"status": "failed", "reason": str(e), "texture_path": None}
