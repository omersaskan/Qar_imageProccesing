"""Abstract base for depth inference providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class DepthProviderBase(ABC):

    name: str = "base"
    license_note: str = ""
    is_experimental: bool = False

    @abstractmethod
    def is_available(self) -> Tuple[bool, str]:
        """Returns (available, reason_if_not)."""

    @abstractmethod
    def infer(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Run depth inference on image_path.
        Returns dict with keys:
          status, depth_map_path, depth_format, model_name, warnings
        """

    def safe_infer(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Wraps infer() — never raises; returns status=unavailable/failed on error."""
        avail, reason = self.is_available()
        if not avail:
            return {
                "status": "unavailable",
                "provider": self.name,
                "reason": reason,
                "depth_map_path": None,
                "depth_format": None,
                "model_name": None,
                "warnings": ["unavailable_model"],
            }
        try:
            return self.infer(image_path, output_dir)
        except Exception as e:
            return {
                "status": "failed",
                "provider": self.name,
                "reason": str(e),
                "depth_map_path": None,
                "depth_format": None,
                "model_name": None,
                "warnings": ["provider_error"],
            }
