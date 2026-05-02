"""Edge cleanup and optional smoothing for raw depth maps."""
from __future__ import annotations

from typing import Dict, Any
import numpy as np


def refine_depth(
    depth: np.ndarray,
    image_rgb: np.ndarray,
    edge_cleanup: bool = True,
    guided_filter: bool = False,
    bilateral: bool = False,
) -> Dict[str, Any]:
    """
    Refine depth map using edge-aware smoothing.
    depth: float32 [0,1], image_rgb: uint8 HxWx3
    Returns dict with 'depth' (refined array) and 'ops_applied' list.
    """
    result = depth.copy()
    ops: list = []

    if edge_cleanup:
        result = _median_edge_cleanup(result)
        ops.append("median_edge_cleanup")

    if bilateral:
        try:
            import cv2
            d_u8 = (result * 255).astype(np.uint8)
            smooth = cv2.bilateralFilter(d_u8, 9, 75, 75)
            result = smooth.astype(np.float32) / 255.0
            ops.append("bilateral_filter")
        except Exception:
            pass

    if guided_filter:
        try:
            result = _guided_filter(result, image_rgb)
            ops.append("guided_filter")
        except Exception:
            pass

    return {"depth": result, "ops_applied": ops}


def _median_edge_cleanup(depth: np.ndarray, ksize: int = 5) -> np.ndarray:
    try:
        import cv2
        d_u16 = (depth * 65535).astype(np.uint16)
        cleaned = cv2.medianBlur(d_u16, ksize)
        return cleaned.astype(np.float32) / 65535.0
    except Exception:
        return depth


def _guided_filter(depth: np.ndarray, guide_rgb: np.ndarray, r: int = 8, eps: float = 0.01) -> np.ndarray:
    """Simple box-filter approximation of guided filter."""
    import cv2
    guide = cv2.cvtColor(guide_rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    d = depth.copy()
    # Mean filter approximation
    mean_d = cv2.boxFilter(d, -1, (r * 2 + 1, r * 2 + 1))
    mean_g = cv2.boxFilter(guide, -1, (r * 2 + 1, r * 2 + 1))
    mean_dg = cv2.boxFilter(d * guide, -1, (r * 2 + 1, r * 2 + 1))
    cov_dg = mean_dg - mean_d * mean_g
    mean_gg = cv2.boxFilter(guide * guide, -1, (r * 2 + 1, r * 2 + 1))
    var_g = mean_gg - mean_g * mean_g
    a = cov_dg / (var_g + eps)
    b = mean_d - a * mean_g
    mean_a = cv2.boxFilter(a, -1, (r * 2 + 1, r * 2 + 1))
    mean_b = cv2.boxFilter(b, -1, (r * 2 + 1, r * 2 + 1))
    return mean_a * guide + mean_b
