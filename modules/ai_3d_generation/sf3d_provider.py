"""
Stable Fast 3D (SF3D) provider for AI 3D generation.

Isolation contract:
  - NEVER imports sf3d, torch, or any heavy ML package in the main process.
  - Uses an isolated subprocess worker (scripts/sf3d_worker.py) executed
    with the SF3D venv Python (DEPTH_PRO_PYTHON_PATH analogue: SF3D_PYTHON_PATH).
  - Safe to import even when SF3D is not installed.

License: Stable Fast 3D / Stability AI — check Stability AI Community License
         before any commercial deployment.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .provider_base import AI3DProviderBase, _unavailable_result, _failed_result

logger = logging.getLogger("ai_3d_generation.sf3d_provider")

_LICENSE_NOTE = (
    "Stable Fast 3D — Stability AI Community License. "
    "Check https://stability.ai/license before commercial use."
)

_STDERR_TAIL_LINES = 20


class SF3DProvider(AI3DProviderBase):

    name = "sf3d"
    license_note = _LICENSE_NOTE
    is_experimental = True
    supports_video_directly = False
    output_format = "glb"

    def __init__(self):
        from modules.operations.settings import settings
        self._settings = settings

    # ── availability ──────────────────────────────────────────────────────────

    def is_available(self) -> Tuple[bool, str]:
        s = self._settings
        if not s.sf3d_enabled:
            return False, "sf3d_disabled"
        py_path = s.sf3d_python_path
        if not py_path:
            return False, "sf3d_python_missing"
        if not Path(py_path).exists():
            return False, f"sf3d_python_missing:{py_path}"
        worker = s.sf3d_worker_script
        if not worker or not Path(worker).exists():
            return False, "sf3d_worker_missing"
        return True, ""

    # ── generate ─────────────────────────────────────────────────────────────

    def generate(
        self,
        input_image_path: str,
        output_dir: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        s = self._settings
        opts = options or {}

        avail, reason = self.is_available()
        if not avail:
            return _unavailable_result(self.name, self.output_format, reason)

        cmd = [
            str(Path(s.sf3d_python_path).resolve()),
            str(Path(s.sf3d_worker_script).resolve()),
            "--image",              str(Path(input_image_path).resolve()),
            "--output-dir",         str(Path(output_dir).resolve()),
            "--device",             opts.get("device", s.sf3d_device),
            "--input-size",         str(opts.get("input_size", s.sf3d_input_size)),
            "--texture-resolution", str(opts.get("texture_resolution", s.sf3d_texture_resolution)),
            "--remesh",             opts.get("remesh", s.sf3d_remesh),
            "--output-format",      opts.get("output_format", s.sf3d_output_format),
        ]
        if opts.get("dry_run"):
            cmd.append("--dry-run")

        timeout = opts.get("timeout", s.sf3d_timeout_sec)
        logs: list[str] = []
        stderr_lines: list[str] = []

        logger.debug("SF3D worker cmd: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return _error_result(
                "sf3d_worker_timeout",
                f"SF3D worker timed out after {timeout}s",
                self.name, self.output_format,
            )
        except Exception as exc:
            return _error_result(
                "sf3d_worker_failed",
                f"Failed to launch SF3D worker: {exc}",
                self.name, self.output_format,
            )

        stderr_lines = (proc.stderr or "").splitlines()
        logs = stderr_lines[-_STDERR_TAIL_LINES:]

        stdout = (proc.stdout or "").strip()
        if not stdout:
            return _error_result(
                "sf3d_worker_invalid_json",
                f"SF3D worker produced no stdout (exit={proc.returncode})",
                self.name, self.output_format,
                logs=logs,
            )

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            return _error_result(
                "sf3d_worker_invalid_json",
                f"SF3D worker JSON parse error: {exc} — stdout: {stdout[:200]}",
                self.name, self.output_format,
                logs=logs,
            )

        if data.get("status") == "ok":
            output_path = data.get("output_path")
            if output_path and not Path(output_path).exists():
                return _error_result(
                    "sf3d_output_missing",
                    f"SF3D worker reported ok but output not found: {output_path}",
                    self.name, self.output_format,
                    logs=logs,
                )
            return {
                "status": "ok",
                "provider": self.name,
                "model_name": data.get("model_name", "stable-fast-3d"),
                "input_image_path": input_image_path,
                "output_path": output_path,
                "output_format": self.output_format,
                "preview_image_path": data.get("preview_image_path"),
                "logs": logs,
                "warnings": data.get("warnings", []) + ["ai_generated_not_true_scan"],
                "error": None,
                "error_code": None,
                "metadata": data.get("metadata", {}),
            }

        if data.get("status") == "unavailable":
            return _unavailable_result(
                self.name, self.output_format,
                data.get("message", data.get("error_code", "sf3d_package_missing")),
            )

        # Worker returned error / unknown status
        return _error_result(
            data.get("error_code", "sf3d_worker_failed"),
            data.get("message", str(data)),
            self.name, self.output_format,
            logs=logs,
        )


def _error_result(
    error_code: str,
    message: str,
    provider: str,
    output_format: str,
    logs: list | None = None,
) -> Dict[str, Any]:
    return {
        "status": "failed",
        "provider": provider,
        "model_name": None,
        "input_image_path": None,
        "output_path": None,
        "output_format": output_format,
        "preview_image_path": None,
        "logs": logs or [],
        "warnings": [],
        "error": message,
        "error_code": error_code,
        "metadata": {},
    }
