"""Apple Depth Pro provider (experimental)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Tuple

from .depth_provider_base import DepthProviderBase


class DepthProProvider(DepthProviderBase):

    name = "depth_pro"
    license_note = (
        "Depth Pro — Apple Research License. "
        "Non-commercial research use only. "
        "Check license before production deployment."
    )
    is_experimental = True

    def __init__(self, checkpoint: str = "", device: str = "cpu"):
        self.checkpoint = checkpoint or os.environ.get("DEPTH_PRO_CHECKPOINT", "")
        self.device = device

    def is_available(self) -> Tuple[bool, str]:
        from modules.operations.settings import settings
        if not settings.depth_pro_enabled:
            return False, "DEPTH_PRO_ENABLED=false"
        try:
            import depth_pro  # noqa: F401
            return True, ""
        except ImportError:
            return False, "depth_pro package not installed"

    def infer(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        import depth_pro
        import numpy as np
        from PIL import Image

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        depth_path = str(Path(output_dir) / "depth_16.png")

        model, transform = depth_pro.create_model_and_transforms(
            device=self.device,
        )
        model.eval()

        image, _, f_px = depth_pro.load_rgb(image_path)
        prediction = model.infer(transform(image), f_px=f_px)
        depth = prediction["depth"].detach().cpu().numpy()

        d_min, d_max = depth.min(), depth.max()
        depth16 = ((depth - d_min) / (d_max - d_min + 1e-8) * 65535).astype("uint16")

        import cv2
        cv2.imwrite(depth_path, depth16)

        return {
            "status": "ok",
            "provider": self.name,
            "depth_map_path": depth_path,
            "depth_format": "png16",
            "model_name": "apple/depth-pro",
            "warnings": ["experimental_provider"],
        }
