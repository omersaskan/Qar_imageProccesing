"""Write depth maps in 16-bit PNG or EXR formats."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


def write_depth_png16(depth_array: np.ndarray, output_path: str) -> str:
    """Write normalized uint16 PNG. Returns output_path."""
    import cv2
    d = depth_array.astype(np.float64)
    d_min, d_max = d.min(), d.max()
    norm = ((d - d_min) / (d_max - d_min + 1e-8) * 65535).astype(np.uint16)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, norm)
    return output_path


def write_depth_exr(depth_array: np.ndarray, output_path: str) -> str:
    """Write 32-bit float EXR. Returns output_path or raises if OpenEXR unavailable."""
    try:
        import OpenEXR
        import Imath
        d = depth_array.astype(np.float32)
        h, w = d.shape[:2]
        header = OpenEXR.Header(w, h)
        header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        exr = OpenEXR.OutputFile(output_path, header)
        exr.writePixels({"Z": d.tobytes()})
        exr.close()
        return output_path
    except ImportError:
        raise RuntimeError("OpenEXR not installed. Install pyopenexr to use EXR output.")


def write_depth_preview(depth_array: np.ndarray, output_path: str) -> str:
    """Write 8-bit colormapped preview PNG."""
    import cv2
    d = depth_array.astype(np.float64)
    norm = ((d - d.min()) / (d.max() - d.min() + 1e-8) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, colored)
    return output_path


def load_depth_png16(depth_path: str) -> np.ndarray:
    """Load 16-bit PNG depth map, return float32 array in [0,1]."""
    import cv2
    raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    return raw.astype(np.float32) / 65535.0
