"""
GLB/glTF validation — two layers:

1. validate_glb_content(path) — pure-Python structural validator.
   No external tools required. Checks binary header, JSON chunk,
   scenes/nodes/meshes, buffer references and material references.
   Returns the Phase 4D result dict shape.

2. validate_glb(path) — optional Khronos CLI wrapper (Sprint 7).
   Returns GltfValidationReport (CLI binary required).
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


# ── Pure-Python structural GLB validator ─────────────────────────────────────

import struct

_GLB_MAGIC          = 0x46546C67   # "glTF" little-endian
_CHUNK_TYPE_JSON    = 0x4E4F534A   # "JSON"
_CHUNK_TYPE_BIN     = 0x004E4942   # "BIN\0"
_SUPPORTED_VERSIONS = frozenset({2})


def validate_glb_content(glb_path: "str | Path | None") -> Dict[str, Any]:
    """
    Pure-Python structural GLB validator.  Never modifies the file.

    Returns
    -------
    dict with keys:
        enabled          : True
        available        : True
        valid            : bool
        issues           : list[str]   — structural problems (invalid = any issue)
        warnings         : list[str]   — non-fatal observations
        metadata         : dict        — counts extracted from the JSON chunk
        error            : str | None  — set only when reading/parsing fails early
    """
    result: Dict[str, Any] = {
        "enabled":   True,
        "available": True,
        "valid":     False,
        "issues":    [],
        "warnings":  [],
        "metadata":  {},
        "error":     None,
    }

    path = Path(glb_path) if glb_path else None

    # ── Check 1: file exists ──────────────────────────────────────────────────
    if not path or not path.exists():
        result["issues"].append("glb_missing")
        result["error"] = "glb_missing"
        return result

    # ── Check 2: extension ────────────────────────────────────────────────────
    if path.suffix.lower() != ".glb":
        result["issues"].append("unexpected_extension")

    # ── Check 3: readable ─────────────────────────────────────────────────────
    try:
        data = path.read_bytes()
    except OSError:
        result["issues"].append("file_unreadable")
        result["error"] = "file_unreadable"
        return result

    # ── Check 4: minimum header size ─────────────────────────────────────────
    if len(data) < 12:
        result["issues"].append("file_too_small_for_header")
        result["error"] = "file_too_small_for_header"
        return result

    # ── Check 5: magic bytes ──────────────────────────────────────────────────
    magic, version, declared_length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        result["issues"].append("invalid_magic")
        result["error"] = "invalid_magic"
        return result

    # ── Check 6: version ─────────────────────────────────────────────────────
    if version not in _SUPPORTED_VERSIONS:
        result["issues"].append(f"unsupported_version_{version}")
        result["warnings"].append(f"GLB version {version} not in supported set {set(_SUPPORTED_VERSIONS)}")

    # ── Check 7: declared length vs actual ───────────────────────────────────
    actual_size = len(data)
    if declared_length != actual_size:
        result["issues"].append("length_mismatch")
        result["warnings"].append(
            f"Declared length {declared_length} != actual file size {actual_size}"
        )

    # ── Check 8: JSON chunk header ────────────────────────────────────────────
    if len(data) < 20:
        result["issues"].append("file_too_small_for_json_chunk")
        result["error"] = "file_too_small_for_json_chunk"
        return result

    json_chunk_len, json_chunk_type = struct.unpack_from("<II", data, 12)
    if json_chunk_type != _CHUNK_TYPE_JSON:
        result["issues"].append("first_chunk_not_json")
        result["error"] = "first_chunk_not_json"
        return result

    # ── Check 9: JSON chunk bounds ────────────────────────────────────────────
    json_start = 20
    json_end   = json_start + json_chunk_len
    if json_end > len(data):
        result["issues"].append("json_chunk_overflows_file")
        result["error"] = "json_chunk_overflows_file"
        return result

    # ── Check 10: JSON parses ─────────────────────────────────────────────────
    json_bytes = data[json_start:json_end].rstrip(b"\x00 ")
    try:
        gltf: Dict[str, Any] = json.loads(json_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        result["issues"].append("json_parse_failed")
        result["error"] = "json_parse_failed"
        return result

    # ── Check 11: optional binary chunk ──────────────────────────────────────
    if json_end + 8 <= len(data):
        bin_chunk_len_val, bin_chunk_type = struct.unpack_from("<II", data, json_end)
        if bin_chunk_type == _CHUNK_TYPE_BIN:
            bin_end = json_end + 8 + bin_chunk_len_val
            if bin_end > len(data):
                result["issues"].append("bin_chunk_overflows_file")

    # ── Extract metadata ──────────────────────────────────────────────────────
    scenes       = gltf.get("scenes", [])
    nodes        = gltf.get("nodes", [])
    meshes       = gltf.get("meshes", [])
    materials    = gltf.get("materials", [])
    textures     = gltf.get("textures", [])
    images       = gltf.get("images", [])
    buffers      = gltf.get("buffers", [])
    buffer_views = gltf.get("bufferViews", [])
    accessors    = gltf.get("accessors", [])

    result["metadata"] = {
        "version":           version,
        "declared_length":   declared_length,
        "actual_size":       actual_size,
        "scene_count":       len(scenes),
        "node_count":        len(nodes),
        "mesh_count":        len(meshes),
        "material_count":    len(materials),
        "texture_count":     len(textures),
        "image_count":       len(images),
        "buffer_count":      len(buffers),
        "buffer_view_count": len(buffer_views),
        "accessor_count":    len(accessors),
    }

    # ── Check 12: scenes exist ────────────────────────────────────────────────
    if len(scenes) == 0:
        result["issues"].append("no_scenes")

    # ── Check 13: nodes ───────────────────────────────────────────────────────
    if len(nodes) == 0:
        result["warnings"].append("no_nodes")

    # ── Check 14: meshes ─────────────────────────────────────────────────────
    if len(meshes) == 0:
        result["issues"].append("missing_meshes")

    # ── Check 15: buffer view → buffer references ─────────────────────────────
    for i, bv in enumerate(buffer_views):
        buf_idx = bv.get("buffer", -1)
        if not isinstance(buf_idx, int) or buf_idx < 0 or buf_idx >= len(buffers):
            result["issues"].append(f"buffer_view_{i}_invalid_buffer_ref")
            break

    # ── Check 16: accessor → buffer view references ───────────────────────────
    for i, acc in enumerate(accessors):
        if "bufferView" in acc:
            bv_idx = acc["bufferView"]
            if not isinstance(bv_idx, int) or bv_idx < 0 or bv_idx >= len(buffer_views):
                result["issues"].append(f"accessor_{i}_invalid_buffer_view_ref")
                break

    # ── Check 17: material → texture references ───────────────────────────────
    for i, mat in enumerate(materials):
        pbr = mat.get("pbrMetallicRoughness") or {}
        tex_ref = (pbr.get("baseColorTexture") or {}).get("index")
        if tex_ref is not None and (not isinstance(tex_ref, int) or tex_ref >= len(textures)):
            result["issues"].append(f"material_{i}_invalid_texture_ref")
            break

    result["valid"] = len(result["issues"]) == 0
    return result
