"""
Elevation estimator — heuristic per-frame pitch / camera height bucket.

Why heuristic?  At extraction time we don't have COLMAP poses yet — those
arrive after `feature_extractor` + `mapper`.  But we want to gate captures
*before* paying for reconstruction.  So we infer elevation bucket
(low / mid / top) from cheap visual cues:

    1. Vertical position of the masked-product centroid.
       Object near the bottom of frame → camera looking down (top elevation).
       Object near the top → camera looking up (low elevation).
       Object centered → mid.
    2. Mask aspect ratio change vs frame aspect — top-down view tends to
       have wider/shorter mask than side view.

Coarse but useful: the gate just needs to know whether 3 elevation
bands are represented.  Refined elevation comes later from real cameras.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json

import cv2
import numpy as np


# Heuristic thresholds — pitch buckets in normalized vertical centroid space
_TOP_VIEW_CY_MAX = 0.40   # centroid y < 40% (object near top of frame) → looking up = LOW elevation
_LOW_VIEW_CY_MIN = 0.65   # centroid y > 65% (object near bottom) → looking down = TOP elevation


@dataclass
class FrameElevation:
    frame_name: str
    centroid_y_norm: float
    bucket: str           # "low" | "mid" | "top" | "unknown"
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ElevationReport:
    frame_count: int
    per_frame: List[FrameElevation] = field(default_factory=list)
    bucket_counts: Dict[str, int] = field(default_factory=lambda: {"low": 0, "mid": 0, "top": 0, "unknown": 0})
    bucket_ratios: Dict[str, float] = field(default_factory=lambda: {"low": 0.0, "mid": 0.0, "top": 0.0, "unknown": 0.0})
    multi_height_score: float = 0.0  # fraction of {low,mid,top} buckets seen
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "per_frame": [p.to_dict() for p in self.per_frame],
            "bucket_counts": self.bucket_counts,
            "bucket_ratios": self.bucket_ratios,
            "multi_height_score": self.multi_height_score,
            "notes": self.notes,
        }


def _load_mask_meta(masks_dir: Path, frame_name: str) -> Optional[Dict[str, Any]]:
    """Try the existing per-frame metadata side-car JSON written by frame_extractor."""
    stem = Path(frame_name).stem
    cand = masks_dir / f"{stem}.json"
    if not cand.exists():
        return None
    try:
        with open(cand, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_mask_image(masks_dir: Path, frame_name: str) -> Optional[np.ndarray]:
    """Side-car PNG fallback if metadata centroid is missing."""
    name = Path(frame_name).name
    for cand in (masks_dir / f"{name}.png", masks_dir / f"{Path(name).stem}.png"):
        if cand.exists():
            try:
                buf = np.fromfile(str(cand), dtype=np.uint8)
                return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            except Exception:
                continue
    return None


def _bucket_for(cy_norm: float) -> str:
    """Map normalized vertical centroid to elevation bucket."""
    if cy_norm < _TOP_VIEW_CY_MAX:
        return "low"        # object high in frame → camera looking UP → LOW elevation viewpoint
    if cy_norm > _LOW_VIEW_CY_MIN:
        return "top"        # object low in frame → camera looking DOWN → TOP elevation
    return "mid"


def _per_frame_bucket(frame_path: Path, masks_dir: Optional[Path]) -> FrameElevation:
    name = frame_path.name
    cy_norm = 0.5
    confidence = 0.3

    if masks_dir is not None:
        meta = _load_mask_meta(masks_dir, name)
        if meta and isinstance(meta.get("centroid"), dict):
            cy = meta["centroid"].get("y")
            # height stored implicitly: meta has occupancy; we need image height for cy_norm.
            # bbox h or inferring from mask dims.
            h = None
            if "bbox" in meta and isinstance(meta["bbox"], dict):
                # bbox is product-only, not full frame — skip
                pass
            mask_img = _load_mask_image(masks_dir, name)
            if mask_img is not None and cy is not None:
                h = mask_img.shape[0]
                cy_norm = float(cy) / float(h) if h else 0.5
                confidence = 0.7
            elif cy is not None:
                # Fall back to assuming centroid is in normalized space already
                cy_norm = float(cy) if 0 <= cy <= 1 else 0.5
                confidence = 0.4

        if confidence < 0.5:
            mask_img = _load_mask_image(masks_dir, name)
            if mask_img is not None:
                ys, _ = np.where(mask_img > 0)
                if ys.size > 0:
                    cy_norm = float(ys.mean()) / float(mask_img.shape[0])
                    confidence = 0.7

    return FrameElevation(
        frame_name=name,
        centroid_y_norm=float(cy_norm),
        bucket=_bucket_for(cy_norm),
        confidence=float(confidence),
    )


def estimate_elevation_distribution(
    frame_paths: List[str],
    masks_dir: Optional[Path] = None,
) -> ElevationReport:
    """
    For each frame, infer elevation bucket from mask centroid y-position.
    Aggregate counts + multi_height_score (fraction of {low,mid,top} seen).
    """
    rep = ElevationReport(frame_count=len(frame_paths))
    if not frame_paths:
        rep.notes.append("no frames")
        return rep

    for fp in frame_paths:
        rep.per_frame.append(_per_frame_bucket(Path(fp), masks_dir))

    for fe in rep.per_frame:
        rep.bucket_counts[fe.bucket] = rep.bucket_counts.get(fe.bucket, 0) + 1

    total = max(rep.frame_count, 1)
    for k, v in rep.bucket_counts.items():
        rep.bucket_ratios[k] = v / total

    seen = sum(1 for k in ("low", "mid", "top") if rep.bucket_counts.get(k, 0) > 0)
    rep.multi_height_score = seen / 3.0

    if masks_dir is None:
        rep.notes.append("no masks_dir provided — heuristic confidence is low")
    if rep.multi_height_score < 0.34:
        rep.notes.append("only one elevation band detected — re-shoot with low / mid / top sweeps")
    elif rep.multi_height_score < 0.67:
        rep.notes.append("two of three elevation bands seen — third is missing")

    return rep
