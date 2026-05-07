"""
PBR / material / texture audit for GLB files.

audit_pbr_textures(glb_path) -> dict

Reads the GLB JSON chunk only — no heavy deps, no image decode.
"""
from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_GLB_MAGIC = 0x46546C67
_CHUNK_TYPE_JSON = 0x4E4F534A
_CHUNK_TYPE_BIN = 0x004E4942

_MOBILE_MAX_TEXTURES = 4
_TEXTURE_RES_WARN = 2048


def _read_gltf_json(glb_path: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON chunk from a GLB without external deps."""
    try:
        data = Path(glb_path).read_bytes()
        if len(data) < 20:
            return None
        magic, _ver, _declared = struct.unpack_from("<III", data, 0)
        if magic != _GLB_MAGIC:
            return None
        chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
        if chunk_type != _CHUNK_TYPE_JSON:
            return None
        json_end = 20 + chunk_len
        if json_end > len(data):
            return None
        json_bytes = data[20:json_end].rstrip(b"\x00 ")
        return json.loads(json_bytes)
    except Exception:
        return None


def audit_pbr_textures(glb_path: Optional[str]) -> Dict[str, Any]:
    """
    Audit PBR materials and textures from the GLB JSON chunk.

    Returns
    -------
    dict with keys:
        enabled, available, material_count, texture_count, image_count,
        has_base_color, has_metallic_roughness, has_normal, has_occlusion,
        has_emissive, max_texture_resolution, issues, warnings, recommendations
    """
    result: Dict[str, Any] = {
        "enabled": True,
        "available": False,
        "material_count": 0,
        "texture_count": 0,
        "image_count": 0,
        "has_base_color": None,
        "has_metallic_roughness": None,
        "has_normal": None,
        "has_occlusion": None,
        "has_emissive": None,
        "max_texture_resolution": None,
        "issues": [],
        "warnings": [],
        "recommendations": [],
    }

    if not glb_path or not Path(glb_path).exists():
        result["issues"].append("glb_missing")
        return result

    gltf = _read_gltf_json(glb_path)
    if gltf is None:
        result["issues"].append("gltf_json_unreadable")
        return result

    result["available"] = True

    materials: List[Dict[str, Any]] = gltf.get("materials", [])
    textures: List[Dict[str, Any]] = gltf.get("textures", [])
    images: List[Dict[str, Any]] = gltf.get("images", [])

    result["material_count"] = len(materials)
    result["texture_count"] = len(textures)
    result["image_count"] = len(images)

    has_base_color = False
    has_mr = False
    has_normal = False
    has_occlusion = False
    has_emissive = False

    for mat in materials:
        pbr = mat.get("pbrMetallicRoughness") or {}
        if pbr.get("baseColorTexture"):
            has_base_color = True
        if pbr.get("metallicRoughnessTexture"):
            has_mr = True
        if mat.get("normalTexture"):
            has_normal = True
        if mat.get("occlusionTexture"):
            has_occlusion = True
        if mat.get("emissiveTexture"):
            has_emissive = True

    if materials:
        result["has_base_color"] = has_base_color
        result["has_metallic_roughness"] = has_mr
        result["has_normal"] = has_normal
        result["has_occlusion"] = has_occlusion
        result["has_emissive"] = has_emissive

    # Detect max texture resolution from embedded PNG headers
    max_res = _detect_max_resolution(glb_path, gltf)
    result["max_texture_resolution"] = max_res

    # ── Issues and warnings ───────────────────────────────────────────────────
    if len(materials) == 0:
        result["issues"].append("no_materials")
        result["recommendations"].append(
            "No materials found. Model will render without color or texture."
        )

    if materials and not has_base_color:
        result["warnings"].append("missing_base_color_texture")
        result["recommendations"].append(
            "No base color texture found. Model may appear flat/untextured."
        )

    if len(textures) > _MOBILE_MAX_TEXTURES:
        result["warnings"].append("high_texture_count_for_mobile")
        result["recommendations"].append(
            f"Texture count ({len(textures)}) exceeds mobile recommendation "
            f"({_MOBILE_MAX_TEXTURES})."
        )

    if max_res is not None and max_res > _TEXTURE_RES_WARN:
        result["warnings"].append("texture_resolution_2048_plus")
        result["recommendations"].append(
            f"Max detected texture resolution {max_res}px may be too large for mobile AR."
        )

    return result


def _detect_max_resolution(
    glb_path: str,
    gltf: Dict[str, Any],
) -> Optional[int]:
    """
    Try to read PNG dimensions from embedded image data in the BIN chunk.
    Non-destructive; returns None if not readable.
    """
    try:
        data = Path(glb_path).read_bytes()

        # Locate BIN chunk
        if len(data) < 20:
            return None
        chunk_len, _chunk_type = struct.unpack_from("<II", data, 12)
        json_end = 20 + chunk_len
        if json_end + 8 > len(data):
            return None
        bin_chunk_len, bin_chunk_type = struct.unpack_from("<II", data, json_end)
        if bin_chunk_type != _CHUNK_TYPE_BIN:
            return None
        bin_start = json_end + 8

        buffer_views: List[Dict[str, Any]] = gltf.get("bufferViews", [])
        images: List[Dict[str, Any]] = gltf.get("images", [])

        max_res: Optional[int] = None
        for img in images:
            bv_idx = img.get("bufferView")
            if bv_idx is None or not isinstance(bv_idx, int):
                continue
            if bv_idx >= len(buffer_views):
                continue
            bv = buffer_views[bv_idx]
            offset = bv.get("byteOffset", 0)
            # Read only enough for PNG IHDR (24 bytes)
            img_start = bin_start + offset
            img_header = data[img_start: img_start + 24]
            w, h = _png_dimensions(img_header)
            if w and h:
                res = max(w, h)
                if max_res is None or res > max_res:
                    max_res = res

        return max_res
    except Exception:
        return None


def _png_dimensions(header: bytes) -> Tuple[Optional[int], Optional[int]]:
    """Extract width and height from a PNG file header (first 24 bytes)."""
    try:
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
            w = struct.unpack_from(">I", header, 16)[0]
            h = struct.unpack_from(">I", header, 20)[0]
            return w, h
    except Exception:
        pass
    return None, None
