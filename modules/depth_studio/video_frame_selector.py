"""Select the best single frame from a video for depth inference."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def _blur_score(frame) -> float:
    """Laplacian variance as sharpness metric (higher = sharper)."""
    try:
        import cv2
        import numpy as np
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _center_content_score(frame) -> float:
    """Rough metric: variance in center crop (more content → higher score)."""
    try:
        import numpy as np
        h, w = frame.shape[:2]
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
        crop = frame[cy - r:cy + r, cx - r:cx + r]
        return float(crop.std())
    except Exception:
        return 0.0


def select_best_frame(
    video_path: str,
    output_path: str,
    candidate_count: int = 10,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Sample candidate_count evenly-spaced frames, pick the sharpest centred one.
    Saves the winner to output_path as JPEG.
    Returns (ok, reason, meta).
    """
    meta: Dict[str, Any] = {
        "video_path": video_path,
        "selected_frame_path": None,
        "selected_frame_index": None,
        "blur_score": None,
        "total_frames": None,
    }
    try:
        import cv2
    except ImportError:
        return False, "cv2 not available", meta

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Cannot open video", meta

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta["total_frames"] = total
    if total <= 0:
        cap.release()
        return False, "Video has no frames", meta

    step = max(1, total // (candidate_count + 1))
    candidates = []
    for i in range(1, candidate_count + 1):
        idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            score = _blur_score(frame) + _center_content_score(frame)
            candidates.append((score, idx, frame))
    cap.release()

    if not candidates:
        return False, "No readable frames found", meta

    best_score, best_idx, best_frame = max(candidates, key=lambda x: x[0])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    import cv2 as _cv2
    _cv2.imwrite(output_path, best_frame)

    meta["selected_frame_path"] = output_path
    meta["selected_frame_index"] = best_idx
    meta["blur_score"] = round(best_score, 2)
    return True, "", meta
