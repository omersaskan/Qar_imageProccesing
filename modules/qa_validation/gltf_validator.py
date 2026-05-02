"""
Khronos glTF Validator wrapper — Sprint 7.

Runs the Khronos `gltf_validator` CLI (or binary) and parses the JSON
output to produce a structured validation report.

Install: https://github.com/KhronosGroup/glTF-Validator
  npm install -g gltf-validator   OR   download platform binary.

Gracefully returns status="unavailable" when binary is not found.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class GltfValidationReport:
    status: str                    # ok | warning | error | unavailable | failed
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    generator: Optional[str] = None
    asset_version: Optional[str] = None
    raw_report: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # raw_report can be large — keep only if small
        if self.raw_report and len(str(self.raw_report)) > 8000:
            d["raw_report"] = None
            d["raw_report_truncated"] = True
        return d


def _find_validator() -> Optional[str]:
    override = os.getenv("GLTF_VALIDATOR_BIN")
    if override and Path(override).exists():
        return override
    for name in ("gltf_validator", "gltf-validator"):
        found = shutil.which(name)
        if found:
            return found
    return None


def validate_glb(
    glb_path: "str | Path",
    timeout_seconds: int = 60,
) -> GltfValidationReport:
    """
    Run Khronos glTF Validator on glb_path, return structured report.
    """
    cli = _find_validator()
    if not cli:
        return GltfValidationReport(
            status="unavailable",
            reason="gltf_validator not found; install via npm install -g gltf-validator",
        )

    glb_path = Path(glb_path)
    if not glb_path.exists():
        return GltfValidationReport(status="failed", reason=f"file not found: {glb_path}")

    try:
        result = subprocess.run(
            [cli, "--output-format", "json", str(glb_path)],
            capture_output=True,
            timeout=timeout_seconds,
            text=True,
        )
        raw_output = (result.stdout or result.stderr or "").strip()
        if not raw_output:
            return GltfValidationReport(
                status="failed",
                reason=f"validator produced no output (exit {result.returncode})",
            )

        report_json: Dict[str, Any] = json.loads(raw_output)
        issues_node = report_json.get("issues", {})
        messages = issues_node.get("messages", [])
        num_errors = issues_node.get("numErrors", 0)
        num_warnings = issues_node.get("numWarnings", 0)
        num_infos = issues_node.get("numInfos", 0)
        asset = report_json.get("info", {})
        generator = asset.get("generator")
        asset_version = asset.get("version")

        if num_errors > 0:
            status = "error"
        elif num_warnings > 0:
            status = "warning"
        else:
            status = "ok"

        return GltfValidationReport(
            status=status,
            error_count=num_errors,
            warning_count=num_warnings,
            info_count=num_infos,
            issues=messages[:50],  # cap to avoid huge manifests
            generator=generator,
            asset_version=asset_version,
            raw_report=report_json,
        )

    except subprocess.TimeoutExpired:
        return GltfValidationReport(
            status="failed",
            reason=f"validator timed out after {timeout_seconds}s",
        )
    except json.JSONDecodeError as exc:
        return GltfValidationReport(
            status="failed",
            reason=f"failed to parse validator JSON output: {exc}",
        )
    except Exception as exc:
        log.warning(f"gltf_validator: {exc}")
        return GltfValidationReport(
            status="failed",
            reason=str(exc)[:300],
        )
