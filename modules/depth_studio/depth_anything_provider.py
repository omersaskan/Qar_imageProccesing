"""Depth Anything V2 provider."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Tuple

from .depth_provider_base import DepthProviderBase


class DepthAnythingV2Provider(DepthProviderBase):

    name = "depth_anything_v2"
    license_note = "Depth Anything V2 — Apache 2.0"
    is_experimental = False

    def __init__(self, checkpoint: str = "", device: str = "cpu"):
        self.checkpoint = checkpoint or os.environ.get("DEPTH_ANYTHING_CHECKPOINT", "")
        self.device = device

    def is_available(self) -> Tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "torch not installed"
        try:
            from depth_anything_v2.dpt import DepthAnythingV2  # noqa: F401
            return True, ""
        except ImportError:
            pass
        # Fallback: check if transformers pipeline is usable
        try:
            from transformers import pipeline  # noqa: F401
            return True, ""
        except ImportError:
            return False, "depth_anything_v2 / transformers not installed"

    def infer(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        depth_path = str(Path(output_dir) / "depth_16.png")

        # Try native depth_anything_v2 first
        try:
            return self._infer_native(image_path, depth_path)
        except Exception:
            pass

        # Fallback: transformers pipeline (depth-estimation)
        return self._infer_transformers(image_path, depth_path)

    def _infer_native(self, image_path: str, depth_path: str) -> Dict[str, Any]:
        import torch
        import cv2
        import numpy as np
        from depth_anything_v2.dpt import DepthAnythingV2 as DAV2

        encoder = "vitl"
        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        model = DAV2(**model_configs[encoder])
        if self.checkpoint and Path(self.checkpoint).exists():
            model.load_state_dict(torch.load(self.checkpoint, map_location="cpu"))
        model = model.to(self.device).eval()

        raw = cv2.imread(image_path)
        depth = model.infer_image(raw)
        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 65535).astype(np.uint16)
        cv2.imwrite(depth_path, depth_norm)

        return {
            "status": "ok",
            "provider": self.name,
            "depth_map_path": depth_path,
            "depth_format": "png16",
            "model_name": f"depth_anything_v2_{encoder}",
            "warnings": [],
        }

    def _infer_transformers(self, image_path: str, depth_path: str) -> Dict[str, Any]:
        import numpy as np
        from PIL import Image
        from transformers import pipeline as hf_pipeline

        pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if self.device == "cuda" else -1,
        )
        img = Image.open(image_path).convert("RGB")
        result = pipe(img)
        depth_arr = np.array(result["depth"])

        # Normalize to 16-bit
        d_min, d_max = depth_arr.min(), depth_arr.max()
        depth16 = ((depth_arr - d_min) / (d_max - d_min + 1e-8) * 65535).astype(np.uint16)

        import cv2
        cv2.imwrite(depth_path, depth16)

        return {
            "status": "ok",
            "provider": self.name,
            "depth_map_path": depth_path,
            "depth_format": "png16",
            "model_name": "depth-anything/Depth-Anything-V2-Small-hf",
            "warnings": [],
        }
