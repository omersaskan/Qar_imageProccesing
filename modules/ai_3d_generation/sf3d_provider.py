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
  - Single-job GPU guard: concurrent calls return status=busy.

License: Stable Fast 3D / Stability AI — check Stability AI Community License
         before any commercial deployment.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
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

# ── single-job GPU guard ──────────────────────────────────────────────────────
# Prevents two simultaneous inference calls competing for VRAM.
# _sf3d_lock is always released in a try/finally — no deadlock risk.
_sf3d_lock = threading.Lock()


# ── path conversion helpers ───────────────────────────────────────────────────

def _windows_to_wsl_path(path_str: str) -> str:
    """
    Convert a Windows absolute path to WSL2 /mnt/X/... format.

    'C:\\Users\\Foo\\bar.png'   → '/mnt/c/Users/Foo/bar.png'
    'C:\\My Files\\out'         → '/mnt/c/My Files/out'
    Already-POSIX paths ('/...') are returned unchanged.
    Relative paths are resolved to absolute via Path.resolve() first.
    UNC paths ('\\\\server\\...') are returned unchanged with a warning.
    Empty string is returned unchanged.
    """
    if not path_str:
        return path_str
    s = str(path_str)

    if s.startswith('/'):
        return s                          # already a POSIX / WSL path

    # UNC paths — not supported; return unchanged
    if s.startswith('\\\\') or s.startswith('//'):
        logger.warning("_windows_to_wsl_path: UNC path unsupported, returning unchanged: %s", s)
        return s

    try:
        p = PureWindowsPath(s)
        drive = p.drive.rstrip(':').lower()   # 'C:' → 'c', '' for relative

        if not drive:
            # Relative path — resolve to absolute using the real filesystem
            try:
                abs_path = str(Path(s).resolve())
                p = PureWindowsPath(abs_path)
                drive = p.drive.rstrip(':').lower()
            except Exception:
                pass

        if not drive:
            # Still no drive (e.g., running on Linux), return unchanged
            logger.warning("_windows_to_wsl_path: no drive letter found for: %s", s)
            return s

        rest = '/'.join(p.parts[1:])          # skip the drive part
        return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"
    except Exception:
        return s                              # best-effort: return unchanged


def _wsl_to_windows_path(path_str: str) -> str:
    """
    Convert a WSL2 /mnt/X/... path to Windows C:\\... format.

    '/mnt/c/Users/Foo/bar.glb'  → 'C:\\Users\\Foo\\bar.glb'
    '/mnt/c'                    → 'C:\\'
    '/mnt/c/'                   → 'C:\\'
    '/tmp/sf3d/out.glb'         → '/tmp/sf3d/out.glb'  (unchanged)
    Paths not starting with '/mnt/' are returned unchanged.
    """
    if not path_str or not path_str.startswith('/mnt/'):
        return path_str

    without_mnt = path_str[5:].rstrip('/')    # 'c' | 'c/Users/...'
    if not without_mnt:
        return path_str                        # bare '/mnt/' — unchanged

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

    # ── WSL2 preflight ────────────────────────────────────────────────────────

    def preflight_wsl(self) -> Dict[str, Any]:
        """
        Fast preflight check for wsl_subprocess mode (no inference, no model load).
        Returns {"ok": bool, "checks": {name: {"ok": bool, "detail": ...}}, ...}
        """
        s = self._settings
        checks: Dict[str, Any] = {}
        overall = True
        distro = getattr(s, "sf3d_wsl_distro", "Ubuntu-24.04")
        wsl_py  = getattr(s, "sf3d_wsl_python_path", "")
        repo_root = getattr(s, "sf3d_wsl_repo_root", "")
        worker_path = f"{repo_root}/scripts/sf3d_worker.py"

        # 1. wsl.exe on PATH?
        wsl_exe = shutil.which("wsl.exe") or shutil.which("wsl")
        checks["wsl_exe"] = {
            "ok": bool(wsl_exe),
            "detail": wsl_exe or "not found on PATH",
        }
        if not wsl_exe:
            return {"ok": False, "checks": checks,
                    "execution_mode": _MODE_WSL, "distro": distro}

        # 2. Distro responds
        try:
            r = subprocess.run(
                ["wsl.exe", "-d", distro, "--", "echo", "preflight_ok"],
                capture_output=True, text=True, timeout=15,
            )
            distro_ok = r.returncode == 0 and "preflight_ok" in r.stdout
        except Exception as e:
            distro_ok = False
            r = type("R", (), {"stderr": str(e), "stdout": ""})()  # type: ignore
        checks["distro"] = {
            "ok": distro_ok,
            "distro": distro,
            "detail": r.stdout.strip() if distro_ok else (r.stderr or "").strip()[:300],
        }
        if not distro_ok:
            overall = False

        # 3. Python interpreter exists
        try:
            r = subprocess.run(
                ["wsl.exe", "-d", distro, "--", "test", "-f", wsl_py],
                capture_output=True, text=True, timeout=10,
            )
            py_ok = r.returncode == 0
        except Exception as e:
            py_ok = False
        checks["python"] = {
            "ok": py_ok,
            "path": wsl_py,
            "detail": "exists" if py_ok else "not found in WSL",
        }
        if not py_ok:
            overall = False

        # 4. Worker script exists
        try:
            r = subprocess.run(
                ["wsl.exe", "-d", distro, "--", "test", "-f", worker_path],
                capture_output=True, text=True, timeout=10,
            )
            worker_ok = r.returncode == 0
        except Exception as e:
            worker_ok = False
        checks["worker_script"] = {
            "ok": worker_ok,
            "path": worker_path,
            "detail": "exists" if worker_ok else "not found in WSL",
        }
        if not worker_ok:
            overall = False

        # 5. Dry-run contract check (fast path — no image needed)
        if py_ok and worker_ok:
            try:
                r = subprocess.run(
                    ["wsl.exe", "-d", distro, "--", wsl_py, worker_path,
                     "--image", "/nonexistent/fake.png",
                     "--output-dir", "/tmp", "--dry-run"],
                    capture_output=True, text=True, timeout=20,
                )
                try:
                    data = json.loads(r.stdout.strip())
                    dry_ok = data.get("dry_run") is True
                except Exception:
                    data, dry_ok = {}, False
                checks["dry_run_contract"] = {
                    "ok": dry_ok,
                    "status": data.get("status") if isinstance(data, dict) else None,
                    "detail": data if dry_ok else r.stdout[:300],
                }
            except Exception as e:
                checks["dry_run_contract"] = {"ok": False, "detail": str(e)}
        else:
            checks["dry_run_contract"] = {
                "ok": False, "detail": "skipped — python or worker not available",
            }

        return {
            "ok": overall,
            "checks": checks,
            "execution_mode": _MODE_WSL,
            "distro": distro,
        }

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

        # Single-job GPU guard
        acquired = _sf3d_lock.acquire(blocking=False)
        if not acquired:
            return {
                "status": "busy",
                "provider": self.name,
                "model_name": None,
                "input_image_path": input_image_path,
                "output_path": None,
                "output_format": self.output_format,
                "preview_image_path": None,
                "logs": [],
                "warnings": [],
                "error": "SF3D GPU job already running — try again when the current job finishes",
                "error_code": "sf3d_job_already_running",
                "metadata": {},
            }

        try:
            mode = getattr(s, "sf3d_execution_mode", _MODE_DISABLED)
            if mode == _MODE_LOCAL_WINDOWS:
                return self._generate_local_windows(input_image_path, output_dir, opts)
            if mode == _MODE_WSL:
                return self._generate_wsl_subprocess(input_image_path, output_dir, opts)
            return _unavailable_result(
                self.name, self.output_format,
                f"sf3d_unknown_execution_mode:{mode}",
            )
        finally:
            _sf3d_lock.release()

    # ── local_windows (original behaviour) ───────────────────────────────────

    def _generate_local_windows(
        self,
        input_image_path: str,
        output_dir: str,
        opts: Dict[str, Any],
    ) -> Dict[str, Any]:
        s = self._settings
        # Ensure absolute paths before passing to subprocess
        img_abs = str(Path(input_image_path).resolve())
        out_abs = str(Path(output_dir).resolve())
        cmd = [
            str(Path(s.sf3d_python_path).resolve()),
            str(Path(s.sf3d_worker_script).resolve()),
            "--image",              img_abs,
            "--output-dir",         out_abs,
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
        # Resolve to absolute Windows paths before converting to WSL
        img_abs    = str(Path(input_image_path).resolve())
        out_abs    = str(Path(output_dir).resolve())
        wsl_image  = _windows_to_wsl_path(img_abs)
        wsl_outdir = _windows_to_wsl_path(out_abs)
        repo_root  = getattr(s, "sf3d_wsl_repo_root", "")
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
