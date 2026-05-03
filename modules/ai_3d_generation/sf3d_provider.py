"""
Stable Fast 3D (SF3D) provider for AI 3D generation.

Isolation contract:
  - NEVER imports sf3d, torch, or any heavy ML package in the main process.
  - Uses an isolated subprocess worker (scripts/sf3d_worker.py) via one of:
      disabled      : always returns unavailable (default — safe)
      local_windows : SF3D venv Python on Windows (requires MSVC / NVCC)
      wsl_subprocess: wsl.exe → Ubuntu-24.04 with CUDA 12.8 + PyTorch cu128
      remote_http   : stub for future remote inference (not yet implemented)
  - Safe to import even when SF3D is not installed.

License: Stable Fast 3D / Stability AI — check Stability AI Community License
         before any commercial deployment.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path, PureWindowsPath
from typing import Dict, Any, Optional, Tuple

from .provider_base import AI3DProviderBase, _unavailable_result, _failed_result

logger = logging.getLogger("ai_3d_generation.sf3d_provider")

_LICENSE_NOTE = (
    "Stable Fast 3D — Stability AI Community License. "
    "Check https://stability.ai/license before commercial use."
)

_STDERR_TAIL_LINES = 20

# Execution mode constants
_MODE_DISABLED      = "disabled"
_MODE_LOCAL_WINDOWS = "local_windows"
_MODE_WSL           = "wsl_subprocess"
_MODE_REMOTE        = "remote_http"


# ── path conversion helpers ───────────────────────────────────────────────────

def _windows_to_wsl_path(path_str: str) -> str:
    """
    Convert an absolute Windows path to WSL2 /mnt/X/... format.
    'C:\\Users\\Foo\\bar.png'  → '/mnt/c/Users/Foo/bar.png'
    'C:\\My Files\\out'        → '/mnt/c/My Files/out'
    Already-POSIX paths (starting with '/') are returned unchanged.
    """
    if not path_str:
        return path_str
    s = str(path_str)
    if s.startswith('/'):
        return s                            # already a POSIX / WSL path
    try:
        p = PureWindowsPath(s)
        drive = p.drive.rstrip(':').lower() # 'C:' → 'c'
        rest  = '/'.join(p.parts[1:])       # skip the drive part
        return f"/mnt/{drive}/{rest}"
    except Exception:
        return s                            # best-effort: return unchanged


def _wsl_to_windows_path(path_str: str) -> str:
    """
    Convert a WSL2 /mnt/X/... path to Windows C:\\... format.
    '/mnt/c/Users/Foo/bar.glb' → 'C:\\Users\\Foo\\bar.glb'
    Paths not starting with '/mnt/' are returned unchanged.
    """
    if not path_str or not path_str.startswith('/mnt/'):
        return path_str
    without_mnt = path_str[5:]             # 'c/...'
    slash_idx = without_mnt.find('/')
    if slash_idx == -1:
        drive = without_mnt.upper()
        rest  = ''
    else:
        drive = without_mnt[:slash_idx].upper()
        rest  = without_mnt[slash_idx + 1:].replace('/', '\\')
    return f"{drive}:\\{rest}" if rest else f"{drive}:\\"


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

        mode = getattr(s, "sf3d_execution_mode", _MODE_DISABLED)

        if not mode or mode == _MODE_DISABLED:
            return False, "sf3d_execution_mode_disabled"

        if mode == _MODE_LOCAL_WINDOWS:
            py_path = s.sf3d_python_path
            if not py_path:
                return False, "sf3d_python_missing"
            if not Path(py_path).exists():
                return False, f"sf3d_python_missing:{py_path}"
            worker = s.sf3d_worker_script
            if not worker or not Path(worker).exists():
                return False, "sf3d_worker_missing"
            return True, ""

        if mode == _MODE_WSL:
            wsl_py = getattr(s, "sf3d_wsl_python_path", "")
            if not wsl_py:
                return False, "sf3d_wsl_python_missing"
            wsl_distro = getattr(s, "sf3d_wsl_distro", "")
            if not wsl_distro:
                return False, "sf3d_wsl_distro_missing"
            repo_root = getattr(s, "sf3d_wsl_repo_root", "")
            if not repo_root:
                return False, "sf3d_wsl_repo_root_missing"
            # Verify worker script is reachable via Windows FS
            worker_wsl = f"{repo_root}/scripts/sf3d_worker.py"
            worker_win = _wsl_to_windows_path(worker_wsl)
            if worker_win and not Path(worker_win).exists():
                return False, f"sf3d_worker_missing:{worker_wsl}"
            return True, ""

        if mode == _MODE_REMOTE:
            return False, "sf3d_remote_http_not_implemented"

        return False, f"sf3d_unknown_execution_mode:{mode}"

    # ── generate dispatcher ───────────────────────────────────────────────────

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

        mode = getattr(s, "sf3d_execution_mode", _MODE_DISABLED)

        if mode == _MODE_LOCAL_WINDOWS:
            return self._generate_local_windows(input_image_path, output_dir, opts)

        if mode == _MODE_WSL:
            return self._generate_wsl_subprocess(input_image_path, output_dir, opts)

        return _unavailable_result(
            self.name, self.output_format,
            f"sf3d_unknown_execution_mode:{mode}",
        )

    # ── local_windows (original behaviour) ───────────────────────────────────

    def _generate_local_windows(
        self,
        input_image_path: str,
        output_dir: str,
        opts: Dict[str, Any],
    ) -> Dict[str, Any]:
        s = self._settings
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
        logger.debug("SF3D local_windows cmd: %s", " ".join(cmd))
        return self._run_worker(cmd, timeout, input_image_path, normalize_path=False)

    # ── wsl_subprocess ────────────────────────────────────────────────────────

    def _generate_wsl_subprocess(
        self,
        input_image_path: str,
        output_dir: str,
        opts: Dict[str, Any],
    ) -> Dict[str, Any]:
        s = self._settings
        wsl_image   = _windows_to_wsl_path(input_image_path)
        wsl_outdir  = _windows_to_wsl_path(output_dir)
        repo_root   = getattr(s, "sf3d_wsl_repo_root", "")
        worker_path = f"{repo_root}/scripts/sf3d_worker.py"

        cmd = [
            "wsl.exe",
            "-d", getattr(s, "sf3d_wsl_distro", "Ubuntu-24.04"),
            "--",
            getattr(s, "sf3d_wsl_python_path", "/home/lenovo/sf3d_venv/bin/python"),
            worker_path,
            "--image",              wsl_image,
            "--output-dir",         wsl_outdir,
            "--device",             opts.get("device", s.sf3d_device),
            "--input-size",         str(opts.get("input_size", s.sf3d_input_size)),
            "--texture-resolution", str(opts.get("texture_resolution", s.sf3d_texture_resolution)),
            "--remesh",             opts.get("remesh", s.sf3d_remesh),
            "--output-format",      opts.get("output_format", s.sf3d_output_format),
        ]
        if opts.get("no_remove_bg"):
            cmd.append("--no-remove-bg")
        if opts.get("dry_run"):
            cmd.append("--dry-run")

        timeout = opts.get("timeout", getattr(s, "sf3d_wsl_timeout_sec", 600))
        logger.debug("SF3D wsl_subprocess cmd: %s", " ".join(cmd))
        return self._run_worker(cmd, timeout, input_image_path, normalize_path=True)

    # ── shared worker runner ──────────────────────────────────────────────────

    def _run_worker(
        self,
        cmd: list,
        timeout: int,
        input_image_path: str,
        normalize_path: bool,
    ) -> Dict[str, Any]:
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

        return self._build_result(data, input_image_path, logs, normalize_path)

    # ── result builder ────────────────────────────────────────────────────────

    def _build_result(
        self,
        data: Dict[str, Any],
        input_image_path: str,
        logs: list,
        normalize_path: bool,
    ) -> Dict[str, Any]:
        execution_mode = getattr(self._settings, "sf3d_execution_mode", _MODE_DISABLED)

        if data.get("status") == "ok":
            output_path = data.get("output_path")
            # WSL paths like /mnt/c/... → Windows C:\...
            if normalize_path and output_path:
                output_path = _wsl_to_windows_path(output_path)
            if output_path and not Path(output_path).exists():
                return _error_result(
                    "sf3d_output_missing",
                    f"SF3D worker reported ok but output not found: {output_path}",
                    self.name, self.output_format,
                    logs=logs,
                )
            metadata = dict(data.get("metadata") or {})
            metadata["execution_mode"] = execution_mode
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
                "metadata": metadata,
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
