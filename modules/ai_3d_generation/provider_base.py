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


class AI3DRemoteAsyncProviderBase(AI3DProviderBase):
    """
    Base for providers that use an asynchronous remote API (Task -> Poll -> Download).
    Provides a blocking generate() implementation that performs the polling loop.
    """

    poll_interval_sec: float = 5.0
    max_poll_attempts: int = 120  # ~10 minutes by default at 5s intervals

    @abstractmethod
    def create_task(
        self,
        input_image_path: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Start the remote task.
        Returns (task_id: str | None, error: str | None).
        """

    @abstractmethod
    def poll_status(self, task_id: str) -> Tuple[str, Optional[str], float]:
        """
        Check task status.
        Returns (status: str, error: str | None, progress: float 0-1).
        
        Statuses should include: 'pending', 'processing', 'succeeded', 'failed', 'cancelled'.
        """

    @abstractmethod
    def download_result(self, task_id: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Download the final GLB/asset.
        Returns (local_path: str | None, error: str | None).
        """

    def generate(
        self,
        input_image_path: str,
        output_dir: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import time
        from pathlib import Path

        _t_start = time.monotonic()
        
        def _with_meta(res: Dict[str, Any], t_id: Optional[str] = None, st: Optional[str] = None, att: int = 0):
            res["metadata"].update({
                "external_provider": True,
                "external_provider_name": self.name,
                "external_task_id": t_id,
                "external_status": st,
                "provider_latency_sec": round(time.monotonic() - _t_start, 2),
                "provider_poll_count": att,
                "privacy_notice": getattr(self, "privacy_notice", None) or "External provider terms of service apply.",
            })
            return res

        # 1. Create Task
        task_id, err = self.create_task(input_image_path, options)
        if err or not task_id:
            return _with_meta(_failed_result(self.name, self.output_format, err or "Task creation failed"))

        # 2. Poll
        status = "pending"
        attempts = 0
        error_msg = None
        
        while attempts < self.max_poll_attempts:
            attempts += 1
            status, error_msg, progress = self.poll_status(task_id)
            if status == "succeeded":
                break
            if status in ("failed", "cancelled"):
                return _with_meta(_failed_result(
                    self.name, self.output_format, 
                    error_msg or f"Task {status}", 
                    error_code=f"provider_task_{status}"
                ), task_id, status, attempts)
            
            time.sleep(self.poll_interval_sec)
        else:
            return _with_meta(_failed_result(self.name, self.output_format, "Polling timed out", error_code="provider_timeout"), task_id, "timeout", attempts)

        # 3. Download
        local_path, err = self.download_result(task_id, output_dir)
        if err or not local_path:
            return _with_meta(_failed_result(self.name, self.output_format, err or "Download failed"), task_id, status, attempts)

        # 4. Success result
        res = _base_result(self.name, self.output_format)
        res["status"] = "ok"
        res["input_image_path"] = input_image_path
        res["output_path"] = local_path
        
        # Strengthened metadata (Phase 1.5)
        _with_meta(res, task_id, status, attempts)
        res["metadata"]["downloaded_output_glb_path"] = local_path
        return res




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
    from .sanitization import sanitize_external_provider_error
    r = _base_result(provider, output_format)
    r["status"] = "unavailable"
    r["error"] = sanitize_external_provider_error(reason)
    r["error_code"] = "provider_unavailable"
    return r


def _failed_result(provider: str, output_format: str, reason: str,
                   error_code: str = "provider_exception") -> Dict[str, Any]:
    from .sanitization import sanitize_external_provider_error
    r = _base_result(provider, output_format)
    r["status"] = "failed"
    r["error"] = sanitize_external_provider_error(reason)
    r["error_code"] = error_code
    
    # Ensure sanitized_error is in metadata for Phase 1.5
    r["metadata"]["sanitized_error"] = r["error"]
    return r



_KNOWN_STATUSES = ("ok", "unavailable", "failed", "disabled", "busy")


def _normalise_status(result: Dict[str, Any], provider: str, output_format: str) -> Dict[str, Any]:
    """Map non-standard status strings to 'failed', preserving original in error."""
    status = result.get("status")
    if status not in _KNOWN_STATUSES:
        result = dict(result)
        original = result.get("error") or result.get("error_code") or status
        from .sanitization import sanitize_text
        result["error"] = sanitize_text(str(original))
        result["status"] = "failed"
    return result
