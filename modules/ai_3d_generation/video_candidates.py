"""
Phase 1 — Video frame candidate selection.

Pure helper module. Does NOT call SF3D or any inference provider.
Uses cv2 (already available in the project) for frame extraction and
sharpness scoring.

Responsibilities:
  - Extract frames from a video at evenly-spaced intervals.
  - Score each frame by sharpness (variance of Laplacian).
  - Return the top-k sharpest frames as candidate source images.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("ai_3d_generation.video_candidates")


def _frame_sharpness(frame) -> float:
    """Compute sharpness score via variance of Laplacian (higher = sharper)."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_top_k_frames(
    video_path: str,
    out_dir: str,
    top_k: int = 5,
    min_spacing_sec: float = 0.4,
) -> List[str]:
    """
    Extract evenly-spaced frames from *video_path*, score each by sharpness,
    and save the top-k sharpest frames to *out_dir*.

    Returns a list of absolute paths to the saved frame images (JPEG),
    ordered by descending sharpness score.

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    out_dir : str
        Directory to write candidate frame images into.
    top_k : int
        Maximum number of frames to return.
    min_spacing_sec : float
        Minimum interval between sampled frames (seconds).

    Returns
    -------
    List[str]
        Paths to saved candidate frames. May be fewer than *top_k* if the
        video is very short.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    if duration_sec <= 0 or total_frames <= 0:
        logger.warning("Video has no frames or zero duration: %s", video_path)
        cap.release()
        return []

    # Determine sampling interval (in frames)
    min_spacing_frames = max(1, int(min_spacing_sec * fps))

    # Sample at regular intervals, never closer than min_spacing
    # We want to sample broadly, then pick top-k by sharpness
    sample_count = max(top_k * 3, 15)  # over-sample to have choice
    step = max(min_spacing_frames, total_frames // sample_count)

    scored_frames: List[Tuple[float, int]] = []  # (sharpness, frame_idx)

    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            frame_idx += step
            continue
        score = _frame_sharpness(frame)
        scored_frames.append((score, frame_idx))
        frame_idx += step

    cap.release()

    if not scored_frames:
        logger.warning("No frames could be read from video: %s", video_path)
        return []

    # Sort by sharpness descending, pick top-k
    scored_frames.sort(key=lambda x: x[0], reverse=True)
    selected = scored_frames[:top_k]

    # Re-sort by frame index for temporal ordering in output names
    selected.sort(key=lambda x: x[1])

    # Save selected frames
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    cap = cv2.VideoCapture(video_path)
    for i, (score, fidx) in enumerate(selected):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f"frame_{i+1:03d}.jpg"
        fpath = out_dir_p / fname
        cv2.imwrite(str(fpath), frame)
        saved_paths.append(str(fpath.resolve()))
        logger.debug("Saved frame %d (idx=%d, sharpness=%.1f) → %s", i+1, fidx, score, fpath)

    cap.release()
    logger.info("Selected %d/%d candidate frames from %s", len(saved_paths), top_k, video_path)
    return saved_paths
