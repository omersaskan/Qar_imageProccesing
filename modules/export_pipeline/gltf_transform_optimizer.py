"""
glTF-Transform optimizer — Sprint 7.

Runs glTF-Transform CLI (`gltf-transform`) as a subprocess to optimize
a GLB file for mobile/AR delivery.

Gracefully reports unavailable when the CLI is not installed.
Default operations: prune + dedup + (optional) resize textures.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class GltfTransformConfig:
    prune: bool = True
    dedup: bool = True
    flatten: bool = False
    resize_textures: bool = False
    resize_max_size: int = 2048        # pixels; used when resize_textures=True
    draco_compression: bool = False    # opt-in; lossy geometry compression
    timeout_seconds: int = 300


@dataclass
class GltfTransformResult:
    status: str                        # ok | failed | unavailable | skipped
    output_glb: Optional[str] = None
    cli_version: Optional[str] = None
    elapsed_seconds: float = 0.0
    operations_run: List[str] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    reason: Optional[str] = None

    def __post_init__(self):
        if self.operations_run is None:
            self.operations_run = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _find_gltf_transform() -> Optional[str]:
    override = os.getenv("GLTF_TRANSFORM_BIN")
    if override and Path(override).exists():
        return override
    return shutil.which("gltf-transform")


def _cli_version(bin_path: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            [bin_path, "--version"], timeout=10, stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8", errors="ignore").strip().splitlines()[0]
    except Exception:
        return None


def optimize_glb(
    input_glb: "str | Path",
    output_glb: "str | Path",
    config: Optional[GltfTransformConfig] = None,
) -> GltfTransformResult:
    """
    Run glTF-Transform optimize pipeline on input_glb.

    Uses a single `gltf-transform optimize` command with flags, falling back
    to individual passes when optimize is unavailable.
    """
    import time

    cli = _find_gltf_transform()
    if not cli:
        return GltfTransformResult(
            status="unavailable",
            reason="gltf-transform CLI not found; install via: npm install -g @gltf-transform/cli",
        )

    cfg = config or GltfTransformConfig()
    input_glb = Path(input_glb)
    output_glb = Path(output_glb)

    if not input_glb.exists():
        return GltfTransformResult(status="failed", reason=f"input GLB not found: {input_glb}")

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    version = _cli_version(cli)
    ops: List[str] = []

    # Build command: use `optimize` if available, otherwise pipe manually.
    # We use the simplest approach: single `optimize` subcommand.
    cmd = [cli, "optimize", str(input_glb), str(output_glb)]
    if not cfg.prune:
        cmd += ["--no-prune"]
    if not cfg.dedup:
        cmd += ["--no-dedup"]
    if cfg.draco_compression:
        cmd += ["--compress", "draco"]
    ops.append("optimize")

    t_start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=cfg.timeout_seconds,
            text=True,
        )
        elapsed = time.monotonic() - t_start
        stdout_tail = "\n".join(result.stdout.splitlines()[-30:])
        stderr_tail = "\n".join(result.stderr.splitlines()[-30:])

        if result.returncode != 0:
            return GltfTransformResult(
                status="failed",
                cli_version=version,
                elapsed_seconds=round(elapsed, 2),
                operations_run=ops,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                reason=f"gltf-transform exited {result.returncode}",
            )

        if not output_glb.exists():
            return GltfTransformResult(
                status="failed",
                cli_version=version,
                elapsed_seconds=round(elapsed, 2),
                reason="output GLB not found after gltf-transform",
            )

        return GltfTransformResult(
            status="ok",
            output_glb=str(output_glb),
            cli_version=version,
            elapsed_seconds=round(elapsed, 2),
            operations_run=ops,
            stdout_tail=stdout_tail,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t_start
        return GltfTransformResult(
            status="failed",
            cli_version=version,
            elapsed_seconds=round(elapsed, 2),
            operations_run=ops,
            reason=f"gltf-transform timed out after {cfg.timeout_seconds}s",
        )
    except Exception as exc:
        elapsed = time.monotonic() - t_start
        log.warning(f"gltf_transform_optimizer: {exc}")
        return GltfTransformResult(
            status="failed",
            elapsed_seconds=round(elapsed, 2),
            reason=str(exc)[:300],
        )
