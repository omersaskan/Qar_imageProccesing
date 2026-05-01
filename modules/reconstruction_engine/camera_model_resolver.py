"""
Camera model resolver — pick the right COLMAP camera_model per device.

Why this exists:
    feature_extractor.py historically hard-coded `--ImageReader.camera_model RADIAL`.
    RADIAL handles ~70° HFOV well; for wider phone ultrawides (95–110°) it
    leaves residual radial distortion at frame edges, which destroys feature
    matches there and shrinks the registered_images count by 20–40%.

Resolution priority:
    1. Device database lookup (FocalLength + Make/Model EXIF)
    2. HFOV computation from EXIF FocalLength + sensor width
    3. Filename hint ("ultrawide", "fisheye")
    4. Conservative default: RADIAL

Outputs a CameraModelDecision the adapter passes to ColmapCommandBuilder.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
import math

logger = logging.getLogger(__name__)


class ColmapCameraModel(str, Enum):
    SIMPLE_RADIAL = "SIMPLE_RADIAL"  # 1 distortion param — narrow lens / unknown
    RADIAL = "RADIAL"                # 2 params — typical phone main camera (60–75°)
    OPENCV = "OPENCV"                # 4 params — wide-angle (75–95°)
    OPENCV_FISHEYE = "OPENCV_FISHEYE"  # 4 fisheye params — ultrawide (>95°)


@dataclass
class CameraModelDecision:
    model: ColmapCameraModel
    estimated_hfov_deg: float
    source: str  # "device_db" | "exif_hfov" | "filename_hint" | "default"
    reason: str
    sample_count: int = 0
    devices_seen: List[str] = field(default_factory=list)
    focal_lengths_mm: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["model"] = self.model.value
        return d


# ─── Device DB ──────────────────────────────────────────────────────────────
# Curated for the most common capture devices.  Add rows as field data arrives.
# (make_lower, model_lower_substring, focal_mm_min, focal_mm_max) → ColmapCameraModel
_DEVICE_DB: List[Tuple[str, str, float, float, ColmapCameraModel, str]] = [
    # iPhone 14/15/16 main camera — 26mm equiv, ~73° HFOV
    ("apple", "iphone", 5.0, 7.5, ColmapCameraModel.RADIAL, "iphone main wide ~73° HFOV"),
    # iPhone Pro ultrawide — 13mm equiv, ~120° HFOV
    ("apple", "iphone", 1.5, 3.0, ColmapCameraModel.OPENCV_FISHEYE, "iphone ultrawide ~120° HFOV"),
    # Pixel 7/8 main — 24mm equiv
    ("google", "pixel", 5.0, 8.0, ColmapCameraModel.RADIAL, "pixel main ~75° HFOV"),
    # Pixel ultrawide — 14mm equiv, ~107° HFOV
    ("google", "pixel", 1.5, 3.5, ColmapCameraModel.OPENCV, "pixel ultrawide ~107° HFOV"),
    # Samsung Galaxy main
    ("samsung", "sm-", 5.0, 8.5, ColmapCameraModel.RADIAL, "samsung main ~75° HFOV"),
    # Samsung ultrawide
    ("samsung", "sm-", 1.0, 3.0, ColmapCameraModel.OPENCV, "samsung ultrawide ~95° HFOV"),
    # GoPro / action cams — fisheye dominant
    ("gopro", "", 0.0, 99.0, ColmapCameraModel.OPENCV_FISHEYE, "gopro action cam fisheye"),
    # DSLR / mirrorless — RADIAL is safe
    ("canon", "", 24.0, 999.0, ColmapCameraModel.RADIAL, "canon dslr/mirrorless"),
    ("nikon", "", 24.0, 999.0, ColmapCameraModel.RADIAL, "nikon dslr/mirrorless"),
    ("sony", "ilc", 16.0, 999.0, ColmapCameraModel.RADIAL, "sony alpha"),
]

# 35mm sensor width = 36mm; HFOV (deg) = 2 * atan(W / (2*f))
# We need *equivalent* focal length to compute apparent HFOV.
_SENSOR_WIDTH_35MM = 36.0


def _hfov_from_focal_mm(focal_mm: float, sensor_width_mm: float = _SENSOR_WIDTH_35MM) -> float:
    """Estimate horizontal FOV in degrees, assuming 35mm-equivalent focal."""
    if focal_mm <= 0:
        return 0.0
    return float(2.0 * math.degrees(math.atan(sensor_width_mm / (2.0 * focal_mm))))


def _model_for_hfov(hfov_deg: float) -> ColmapCameraModel:
    """Pick the COLMAP model that best fits the FOV bucket."""
    if hfov_deg <= 0:
        return ColmapCameraModel.RADIAL
    if hfov_deg < 75.0:
        return ColmapCameraModel.RADIAL
    if hfov_deg < 95.0:
        return ColmapCameraModel.OPENCV
    return ColmapCameraModel.OPENCV_FISHEYE


def _read_exif(path: Path) -> Dict[str, Any]:
    """Best-effort EXIF read via Pillow.  Returns {} on failure."""
    try:
        from PIL import Image, ExifTags
        with Image.open(path) as im:
            raw = im.getexif() or {}
        decoded: Dict[str, Any] = {}
        for tag, val in raw.items():
            name = ExifTags.TAGS.get(tag, str(tag))
            decoded[name] = val
        return decoded
    except Exception:
        return {}


def _parse_focal_mm(exif: Dict[str, Any]) -> float:
    """
    Try several EXIF fields to recover the 35mm-equivalent focal length.
    FocalLengthIn35mmFilm > FocalLength + FocalPlaneXResolution > FocalLength alone.
    """
    if not exif:
        return 0.0

    f35 = exif.get("FocalLengthIn35mmFilm")
    if f35:
        try:
            return float(f35)
        except (TypeError, ValueError):
            pass

    f_raw = exif.get("FocalLength")
    if f_raw is not None:
        try:
            # Pillow may return IFDRational
            if hasattr(f_raw, "numerator") and hasattr(f_raw, "denominator"):
                return float(f_raw.numerator) / float(f_raw.denominator or 1)
            if isinstance(f_raw, tuple) and len(f_raw) == 2:
                return float(f_raw[0]) / float(f_raw[1] or 1)
            return float(f_raw)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return 0.0


def _device_db_match(make: str, model: str, focal_mm: float) -> Optional[Tuple[ColmapCameraModel, str]]:
    make_l = (make or "").lower()
    model_l = (model or "").lower()
    for row_make, row_model_sub, fmin, fmax, cm, reason in _DEVICE_DB:
        if row_make and row_make not in make_l:
            continue
        if row_model_sub and row_model_sub not in model_l:
            continue
        if not (fmin <= focal_mm <= fmax):
            continue
        return cm, reason
    return None


def _filename_hint(path: Path) -> Optional[Tuple[ColmapCameraModel, str]]:
    name = path.stem.lower()
    if "ultrawide" in name or "uw_" in name or "fisheye" in name:
        return ColmapCameraModel.OPENCV_FISHEYE, "filename hint: ultrawide/fisheye"
    if "wide" in name and "ultrawide" not in name:
        return ColmapCameraModel.OPENCV, "filename hint: wide"
    return None


def resolve_for_frames(
    frame_paths: List[str],
    max_samples: int = 8,
    fallback: ColmapCameraModel = ColmapCameraModel.RADIAL,
) -> CameraModelDecision:
    """
    Sample up to `max_samples` frames, read EXIF, vote on the best model.

    Voting policy:
        - Device-DB hits override everything.
        - Otherwise majority HFOV bucket wins.
        - Mixed-device captures fall back to OPENCV (4 params, safer for variety).
    """
    if not frame_paths:
        return CameraModelDecision(
            model=fallback,
            estimated_hfov_deg=0.0,
            source="default",
            reason="no frames provided",
        )

    paths = [Path(p) for p in frame_paths]
    paths = [p for p in paths if p.exists() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")]

    if not paths:
        return CameraModelDecision(
            model=fallback,
            estimated_hfov_deg=0.0,
            source="default",
            reason="no readable frames",
        )

    # Even sampling
    if len(paths) > max_samples:
        step = len(paths) / max_samples
        paths = [paths[int(i * step)] for i in range(max_samples)]

    devices: List[str] = []
    focals: List[float] = []
    hfovs: List[float] = []
    db_votes: Dict[ColmapCameraModel, int] = {}
    db_reasons: List[str] = []
    hint_votes: Dict[ColmapCameraModel, int] = {}

    for p in paths:
        exif = _read_exif(p)
        make = str(exif.get("Make", "")).strip()
        model = str(exif.get("Model", "")).strip()
        if make or model:
            devices.append(f"{make} {model}".strip())

        focal = _parse_focal_mm(exif)
        if focal > 0:
            focals.append(focal)
            hfovs.append(_hfov_from_focal_mm(focal))

        db = _device_db_match(make, model, focal) if focal > 0 else None
        if db:
            cm, reason = db
            db_votes[cm] = db_votes.get(cm, 0) + 1
            db_reasons.append(reason)
        else:
            hint = _filename_hint(p)
            if hint:
                hint_votes[hint[0]] = hint_votes.get(hint[0], 0) + 1

    sample_count = len(paths)
    devices_unique = sorted(set(devices))

    # 1. Device DB — strongest signal
    if db_votes:
        winner = max(db_votes.items(), key=lambda kv: kv[1])[0]
        avg_hfov = float(sum(hfovs) / len(hfovs)) if hfovs else 0.0
        reason = f"device_db match {db_votes[winner]}/{sample_count} samples; {db_reasons[0] if db_reasons else ''}"
        if len(set(db_votes.keys())) > 1:
            reason += f" (mixed: {dict(db_votes)} → most-common wins)"
        logger.info(f"[camera_model] device_db → {winner.value} ({reason})")
        return CameraModelDecision(
            model=winner,
            estimated_hfov_deg=avg_hfov,
            source="device_db",
            reason=reason,
            sample_count=sample_count,
            devices_seen=devices_unique,
            focal_lengths_mm=sorted(set(focals)),
        )

    # 2. HFOV computation
    if hfovs:
        avg = float(sum(hfovs) / len(hfovs))
        cm = _model_for_hfov(avg)
        # If HFOV variance is wide (mixed-lens capture), step up to OPENCV
        if hfovs and (max(hfovs) - min(hfovs)) > 30.0:
            cm = ColmapCameraModel.OPENCV
            reason = f"mixed-lens capture (HFOV {min(hfovs):.0f}°–{max(hfovs):.0f}°) → OPENCV"
        else:
            reason = f"avg HFOV {avg:.1f}° from {len(hfovs)}/{sample_count} EXIF samples"
        logger.info(f"[camera_model] exif_hfov → {cm.value} ({reason})")
        return CameraModelDecision(
            model=cm,
            estimated_hfov_deg=avg,
            source="exif_hfov",
            reason=reason,
            sample_count=sample_count,
            devices_seen=devices_unique,
            focal_lengths_mm=sorted(set(focals)),
        )

    # 3. Filename hint
    if hint_votes:
        winner = max(hint_votes.items(), key=lambda kv: kv[1])[0]
        logger.info(f"[camera_model] filename_hint → {winner.value}")
        return CameraModelDecision(
            model=winner,
            estimated_hfov_deg=0.0,
            source="filename_hint",
            reason=f"filename keyword vote {hint_votes[winner]}/{sample_count}",
            sample_count=sample_count,
            devices_seen=devices_unique,
        )

    # 4. Default
    logger.info(f"[camera_model] no signal → default {fallback.value}")
    return CameraModelDecision(
        model=fallback,
        estimated_hfov_deg=0.0,
        source="default",
        reason="no EXIF / DB / hint signal — conservative fallback",
        sample_count=sample_count,
        devices_seen=devices_unique,
    )
