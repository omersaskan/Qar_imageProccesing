"""
COLMAP sparse text-format parser.

Reads cameras.txt and images.txt from a COLMAP sparse model directory.
Both the model-specific sub-directory (e.g. sparse/0/) and the top-level
sparse/ directory are accepted.

Returns lightweight plain-dict structures; no external deps beyond stdlib.
If files are missing or malformed the functions return empty containers and
log a warning — callers must never crash on absent sparse output.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────── cameras.txt ────────────────────────────

def parse_cameras_txt(cameras_txt: Path) -> Dict[int, Dict[str, Any]]:
    """
    Parse COLMAP cameras.txt.

    Returns {camera_id: {model, width, height, params[]}}.
    Empty dict when the file doesn't exist or is unreadable.
    """
    cameras: Dict[int, Dict[str, Any]] = {}
    if not cameras_txt.exists():
        return cameras
    try:
        with open(cameras_txt, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cam_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]
                cameras[cam_id] = {
                    "camera_id": cam_id,
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params,
                }
    except Exception as exc:
        logging.warning(f"colmap_sparse_parser: failed to parse {cameras_txt}: {exc}")
    return cameras


# ──────────────────────────── images.txt ────────────────────────────

def parse_images_txt(images_txt: Path) -> List[Dict[str, Any]]:
    """
    Parse COLMAP images.txt.

    Each registered image has two consecutive lines:
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      POINTS2D[] as (X Y POINT3D_ID) ...

    Returns list of image dicts with qvec, tvec, camera_id, name.
    Empty list on missing / malformed file.
    """
    images: List[Dict[str, Any]] = []
    if not images_txt.exists():
        return images
    try:
        with open(images_txt, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l for l in f if l.strip() and not l.startswith("#")]
        # Lines come in pairs: header + points2d
        i = 0
        while i < len(lines):
            header = lines[i].strip().split()
            i += 2  # skip points2d line
            if len(header) < 10:
                continue
            img_id = int(header[0])
            qvec = [float(header[1]), float(header[2]), float(header[3]), float(header[4])]
            tvec = [float(header[5]), float(header[6]), float(header[7])]
            camera_id = int(header[8])
            name = header[9]
            images.append({
                "image_id": img_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            })
    except Exception as exc:
        logging.warning(f"colmap_sparse_parser: failed to parse {images_txt}: {exc}")
    return images


# ──────────────────────────── directory probe ────────────────────────────

def find_sparse_model_dir(sparse_root: Path) -> Optional[Path]:
    """
    Given a sparse/ root, return the best sub-directory that contains
    cameras.txt.  Prefers the numerically smallest (i.e. model 0).
    Returns sparse_root itself if cameras.txt is directly inside it.
    Returns None if no cameras.txt is found.
    """
    if (sparse_root / "cameras.txt").exists():
        return sparse_root
    # COLMAP writes numbered sub-dirs: 0/, 1/, …
    candidates = sorted(
        [p for p in sparse_root.iterdir() if p.is_dir()],
        key=lambda p: (not p.name.isdigit(), p.name),
    )
    for cand in candidates:
        if (cand / "cameras.txt").exists():
            return cand
    return None


def load_sparse_model(attempt_dir: Path) -> Tuple[
    Dict[int, Dict[str, Any]],
    List[Dict[str, Any]],
    Optional[Path],
]:
    """
    Convenience loader: probe attempt_dir/sparse for cameras.txt + images.txt.

    Returns (cameras, images, model_dir_or_None).
    Both cameras and images are empty containers when sparse output is absent.
    """
    sparse_root = attempt_dir / "sparse"
    if not sparse_root.exists():
        return {}, [], None

    model_dir = find_sparse_model_dir(sparse_root)
    if model_dir is None:
        return {}, [], None

    cameras = parse_cameras_txt(model_dir / "cameras.txt")
    images = parse_images_txt(model_dir / "images.txt")
    return cameras, images, model_dir
