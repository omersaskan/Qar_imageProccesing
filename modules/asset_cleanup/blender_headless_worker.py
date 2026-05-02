"""
Blender headless worker — Sprint 6.

Runs Blender as a subprocess in background mode (`blender -b --python …`)
to normalize mesh geometry and export GLB.

Requires no Blender GUI; gracefully reports unavailable when the blender
binary is not found on PATH or at BLENDER_BIN env override.

Entry point:
    run_blender_cleanup(input_path, output_glb, config) -> BlenderWorkerResult
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .blender_script_generator import generate_cleanup_script
from .mesh_normalization import NormalizationConfig

log = logging.getLogger(__name__)


@dataclass
class BlenderWorkerResult:
    status: str                        # ok | failed | unavailable
    output_glb: Optional[str] = None
    blender_version: Optional[str] = None
    elapsed_seconds: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _find_blender() -> Optional[str]:
    override = os.getenv("BLENDER_BIN")
    if override and Path(override).exists():
        return override
    found = shutil.which("blender")
    return found


def _blender_version(blender_bin: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            [blender_bin, "--version"], timeout=15, stderr=subprocess.DEVNULL
        )
        line = out.decode("utf-8", errors="ignore").splitlines()[0]
        return line.strip()
    except Exception:
        return None


def run_blender_cleanup(
    input_path: "str | Path",
    output_glb: "str | Path",
    config: Optional[NormalizationConfig] = None,
    timeout_seconds: int = 600,
) -> BlenderWorkerResult:
    """
    Run Blender headless cleanup on input_path, write result to output_glb.

    input_path: OBJ / PLY / FBX / GLB accepted by Blender.
    output_glb: destination GLB path (created by the Blender script).
    config: normalization parameters.
    """
    import time

    blender_bin = _find_blender()
    if not blender_bin:
        return BlenderWorkerResult(
            status="unavailable",
            reason="blender binary not found; set BLENDER_BIN or add blender to PATH",
        )

    version = _blender_version(blender_bin)
    input_path = Path(input_path)
    output_glb = Path(output_glb)
    output_glb.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        return BlenderWorkerResult(
            status="failed",
            reason=f"input mesh not found: {input_path}",
        )

    config = config or NormalizationConfig()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as script_file:
        script_path = script_file.name
        script_file.write(
            generate_cleanup_script(
                input_path=str(input_path),
                output_glb=str(output_glb),
                config=config,
            )
        )

    t_start = time.monotonic()
    try:
        cmd = [blender_bin, "-b", "--python", script_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_seconds,
            text=True,
        )
        elapsed = time.monotonic() - t_start

        stdout_tail = "\n".join(result.stdout.splitlines()[-40:])
        stderr_tail = "\n".join(result.stderr.splitlines()[-40:])

        if result.returncode != 0:
            return BlenderWorkerResult(
                status="failed",
                blender_version=version,
                elapsed_seconds=round(elapsed, 2),
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                reason=f"blender exited with code {result.returncode}",
            )

        if not output_glb.exists():
            return BlenderWorkerResult(
                status="failed",
                blender_version=version,
                elapsed_seconds=round(elapsed, 2),
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                reason="blender finished but output GLB not found",
            )

        return BlenderWorkerResult(
            status="ok",
            output_glb=str(output_glb),
            blender_version=version,
            elapsed_seconds=round(elapsed, 2),
            stdout_tail=stdout_tail,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t_start
        return BlenderWorkerResult(
            status="failed",
            blender_version=version,
            elapsed_seconds=round(elapsed, 2),
            reason=f"blender timed out after {timeout_seconds}s",
        )
    except Exception as exc:
        elapsed = time.monotonic() - t_start
        log.warning(f"blender_headless_worker: unexpected error: {exc}")
        return BlenderWorkerResult(
            status="failed",
            elapsed_seconds=round(elapsed, 2),
            reason=str(exc)[:300],
        )
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass
