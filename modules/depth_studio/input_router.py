"""Route uploaded file to image or video path."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}


def route_input(file_path: str) -> Tuple[str, str]:
    """
    Returns (input_type, file_path) where input_type is 'image' or 'video'.
    Raises ValueError for unsupported formats.
    """
    ext = Path(file_path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image", file_path
    if ext in VIDEO_EXTS:
        return "video", file_path
    raise ValueError(f"Unsupported file type: {ext}. Supported: {sorted(IMAGE_EXTS | VIDEO_EXTS)}")


def is_image(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in IMAGE_EXTS


def is_video(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in VIDEO_EXTS
