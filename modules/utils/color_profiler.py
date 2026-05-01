"""
Dynamic product/background color detection for texture normalization
and chromatic leakage isolation.

Replaces hardcoded `EXPECTED_PRODUCT_COLOR=black` and the orange-only
chromatic leakage filter in isolation.py with sampled, per-job decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np


class ColorCategory(str, Enum):
    BLACK = "black"
    DARK = "dark"
    MID = "mid"
    WHITE_CREAM = "white_cream"
    VIBRANT = "vibrant"
    UNKNOWN = "unknown"


# Per-category texture normalization tune.
# brightness_target = histogram p98 hedefi; gamma < 1 gölgeyi kaldırır.
_PROFILE_TABLE: Dict[ColorCategory, Dict[str, float]] = {
    ColorCategory.BLACK:       {"brightness_target": 200.0, "gamma": 0.90, "saturation": 1.10},
    ColorCategory.DARK:        {"brightness_target": 215.0, "gamma": 0.92, "saturation": 1.05},
    ColorCategory.MID:         {"brightness_target": 225.0, "gamma": 0.95, "saturation": 1.05},
    ColorCategory.WHITE_CREAM: {"brightness_target": 240.0, "gamma": 1.00, "saturation": 1.00},
    ColorCategory.VIBRANT:     {"brightness_target": 230.0, "gamma": 0.95, "saturation": 1.15},
    ColorCategory.UNKNOWN:     {"brightness_target": 225.0, "gamma": 0.95, "saturation": 1.05},
}


@dataclass
class ColorProfile:
    category: ColorCategory
    product_rgb: Tuple[int, int, int]
    background_rgb: Tuple[int, int, int]
    brightness_target: float
    gamma: float
    saturation: float
    sample_count: int = 0
    confidence: float = 0.0
    source: str = "detected"  # "detected" | "override" | "fallback"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["product_rgb"] = list(self.product_rgb)
        d["background_rgb"] = list(self.background_rgb)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ColorProfile":
        cat_raw = d.get("category", "unknown")
        try:
            category = ColorCategory(cat_raw)
        except ValueError:
            category = ColorCategory.UNKNOWN
        return cls(
            category=category,
            product_rgb=tuple(d.get("product_rgb", (128, 128, 128))),
            background_rgb=tuple(d.get("background_rgb", (255, 255, 255))),
            brightness_target=float(d.get("brightness_target", 225.0)),
            gamma=float(d.get("gamma", 0.95)),
            saturation=float(d.get("saturation", 1.05)),
            sample_count=int(d.get("sample_count", 0)),
            confidence=float(d.get("confidence", 0.0)),
            source=str(d.get("source", "detected")),
        )

    @classmethod
    def fallback(cls, source: str = "fallback") -> "ColorProfile":
        tune = _PROFILE_TABLE[ColorCategory.UNKNOWN]
        return cls(
            category=ColorCategory.UNKNOWN,
            product_rgb=(128, 128, 128),
            background_rgb=(255, 255, 255),
            brightness_target=tune["brightness_target"],
            gamma=tune["gamma"],
            saturation=tune["saturation"],
            sample_count=0,
            confidence=0.0,
            source=source,
        )

    @classmethod
    def from_override(cls, expected_color: str) -> Optional["ColorProfile"]:
        """Build a profile from a legacy EXPECTED_PRODUCT_COLOR string."""
        try:
            category = ColorCategory(expected_color.lower())
        except ValueError:
            return None
        tune = _PROFILE_TABLE[category]
        product_rgb = {
            ColorCategory.BLACK:       (20, 20, 20),
            ColorCategory.DARK:        (70, 70, 70),
            ColorCategory.MID:         (140, 140, 140),
            ColorCategory.WHITE_CREAM: (235, 230, 220),
            ColorCategory.VIBRANT:     (200, 100, 100),
            ColorCategory.UNKNOWN:     (128, 128, 128),
        }[category]
        return cls(
            category=category,
            product_rgb=product_rgb,
            background_rgb=(255, 255, 255),
            brightness_target=tune["brightness_target"],
            gamma=tune["gamma"],
            saturation=tune["saturation"],
            source="override",
            confidence=1.0,
        )


def _read_bgr(path: Path) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _read_mask(path: Path) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def _classify(value: float, saturation: float) -> ColorCategory:
    if value < 60 and saturation < 60:
        return ColorCategory.BLACK
    if value < 120 and saturation < 90:
        return ColorCategory.DARK
    if value > 205 and saturation < 55:
        return ColorCategory.WHITE_CREAM
    if saturation > 105:
        return ColorCategory.VIBRANT
    return ColorCategory.MID


def _evenly_sampled(items: List[Path], n: int) -> List[Path]:
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def detect_color_profile(
    frame_paths: List[Path],
    masks_dir: Optional[Path] = None,
    max_samples: int = 10,
) -> ColorProfile:
    """
    Sample frames + masks, return a ColorProfile.

    Falls back to a neutral profile if no usable samples (no frames,
    all masks empty, all frames unreadable).
    """
    if not frame_paths:
        return ColorProfile.fallback("no_frames")

    sampled = _evenly_sampled(sorted(frame_paths), max_samples)

    product_pixels: List[np.ndarray] = []
    background_pixels: List[np.ndarray] = []
    samples_with_mask = 0

    for fp in sampled:
        bgr = _read_bgr(fp)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]

        mask: Optional[np.ndarray] = None
        if masks_dir is not None:
            for cand in (masks_dir / f"{fp.name}.png", masks_dir / f"{fp.stem}.png"):
                if cand.exists():
                    mask = _read_mask(cand)
                    if mask is not None and mask.shape[:2] == (h, w):
                        break
                    mask = None

        if mask is not None and np.any(mask > 0):
            samples_with_mask += 1
            product_pixels.append(bgr[mask > 0])
            background_pixels.append(bgr[mask == 0])
        else:
            # Center crop heuristic: middle 40% box = product, outer ring = background
            cy0, cy1 = int(h * 0.30), int(h * 0.70)
            cx0, cx1 = int(w * 0.30), int(w * 0.70)
            product_pixels.append(bgr[cy0:cy1, cx0:cx1].reshape(-1, 3))
            border_mask = np.ones((h, w), dtype=bool)
            border_mask[cy0:cy1, cx0:cx1] = False
            background_pixels.append(bgr[border_mask])

    if not product_pixels:
        return ColorProfile.fallback("no_readable_frames")

    product_bgr = np.concatenate(product_pixels, axis=0)
    background_bgr = np.concatenate(background_pixels, axis=0)

    # Subsample to keep medians fast
    if len(product_bgr) > 200_000:
        idx = np.random.default_rng(seed=0).choice(len(product_bgr), 200_000, replace=False)
        product_bgr = product_bgr[idx]
    if len(background_bgr) > 200_000:
        idx = np.random.default_rng(seed=1).choice(len(background_bgr), 200_000, replace=False)
        background_bgr = background_bgr[idx]

    product_bgr_med = np.median(product_bgr, axis=0).astype(np.uint8)
    background_bgr_med = np.median(background_bgr, axis=0).astype(np.uint8)

    # HSV in OpenCV: H 0-179, S 0-255, V 0-255
    hsv = cv2.cvtColor(product_bgr_med.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
    saturation = float(hsv[1])
    value = float(hsv[2])

    category = _classify(value=value, saturation=saturation)
    tune = _PROFILE_TABLE[category]

    # Confidence: more samples + more mask coverage = higher
    confidence = min(1.0, (samples_with_mask / max(len(sampled), 1)) * 0.6 + 0.4)

    # BGR → RGB for human-readable storage
    product_rgb = (int(product_bgr_med[2]), int(product_bgr_med[1]), int(product_bgr_med[0]))
    background_rgb = (int(background_bgr_med[2]), int(background_bgr_med[1]), int(background_bgr_med[0]))

    return ColorProfile(
        category=category,
        product_rgb=product_rgb,
        background_rgb=background_rgb,
        brightness_target=tune["brightness_target"],
        gamma=tune["gamma"],
        saturation=tune["saturation"],
        sample_count=len(sampled),
        confidence=confidence,
        source="detected",
    )


def resolve_color_profile(
    expected_color_setting: str,
    frame_paths: List[Path],
    masks_dir: Optional[Path] = None,
) -> ColorProfile:
    """
    Top-level entry: respects EXPECTED_PRODUCT_COLOR override, otherwise detects.
    'auto' or 'unknown' → run detect_color_profile.
    Anything else → from_override (legacy behavior preserved).
    """
    setting = (expected_color_setting or "").strip().lower()
    if setting and setting not in ("auto", "unknown", ""):
        override = ColorProfile.from_override(setting)
        if override is not None:
            return override
    try:
        return detect_color_profile(frame_paths, masks_dir)
    except Exception:
        return ColorProfile.fallback("detection_failed")


def background_leakage_mask(
    rgb_samples: np.ndarray,
    background_rgb: Tuple[int, int, int],
    tolerance: int = 35,
) -> np.ndarray:
    """
    Boolean mask of vertices/pixels whose RGB is within `tolerance`
    of the detected background color (Chebyshev / L∞).

    Replaces the hardcoded orange filter (R>140, G>70, B<110, R>G+25)
    with a generic per-job background detector.
    """
    if rgb_samples.ndim != 2 or rgb_samples.shape[1] < 3:
        return np.zeros(len(rgb_samples), dtype=bool)
    bg = np.asarray(background_rgb, dtype=np.int16).reshape(1, 3)
    diff = np.abs(rgb_samples[:, :3].astype(np.int16) - bg)
    return np.max(diff, axis=1) <= tolerance
