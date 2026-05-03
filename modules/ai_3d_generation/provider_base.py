"""Abstract base for AI 3D generation providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class AI3DProviderBase(ABC):

    name: str = "base"
    license_note: str = ""
    is_experimental: bool = True
    supports_video_directly: bool = False
    output_format: str = "glb"

    @abstractmethod
    def is_available(self) -> Tuple[bool, str]:
        """Return (available: bool, reason_if_not: str)."""

    @abstractmethod
    def generate(
        self,
        input_image_path: str,
        output_dir: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run 3D generation from a single image.

        Returns a result dict:
          status            : "ok" | "unavailable" | "failed"
          provider          : str
          model_name        : str | None
          input_image_path  : str
          output_path       : str | None
          output_format     : str
          preview_image_path: str | None
          logs              : list[str]
          warnings          : list[str]
          error             : str | None
          error_code        : str | None
          metadata          : dict
        """

    def safe_generate(
        self,
        input_image_path: str,
        output_dir: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Wraps generate() — never raises.
        Returns status=unavailable/failed on error; normalises non-standard
        status values to "failed" (original status preserved in error field).
        """
        avail, reason = self.is_available()
        if not avail:
            return _unavailable_result(self.name, self.output_format, reason)
        try:
            result = self.generate(input_image_path, output_dir, options)
            return _normalise_status(result, self.name, self.output_format)
        except Exception as exc:
            return _failed_result(self.name, self.output_format, str(exc))


# ── result helpers ────────────────────────────────────────────────────────────

def _base_result(provider: str, output_format: str) -> Dict[str, Any]:
    return {
        "provider": provider,
        "model_name": None,
        "input_image_path": None,
        "output_path": None,
        "output_format": output_format,
        "preview_image_path": None,
        "logs": [],
        "warnings": [],
        "error": None,
        "error_code": None,
        "metadata": {},
    }


def _unavailable_result(provider: str, output_format: str, reason: str) -> Dict[str, Any]:
    r = _base_result(provider, output_format)
    r["status"] = "unavailable"
    r["error"] = reason
    r["error_code"] = "provider_unavailable"
    return r


def _failed_result(provider: str, output_format: str, reason: str,
                   error_code: str = "provider_exception") -> Dict[str, Any]:
    r = _base_result(provider, output_format)
    r["status"] = "failed"
    r["error"] = reason
    r["error_code"] = error_code
    return r


_KNOWN_STATUSES = ("ok", "unavailable", "failed", "disabled")


def _normalise_status(result: Dict[str, Any], provider: str, output_format: str) -> Dict[str, Any]:
    """Map non-standard status strings to 'failed', preserving original in error."""
    status = result.get("status")
    if status not in _KNOWN_STATUSES:
        result = dict(result)
        original = result.get("error") or result.get("error_code") or status
        result["error"] = original
        result["status"] = "failed"
    return result
