"""
License manifest — Sprint 8.

Tracks the license metadata of every tool and asset involved in producing
a 3D asset.  Written as license_manifest.json next to the final asset.

Key concerns:
  - OpenMVS is AGPL-3.0 — if distributed, source must be offered.
  - COLMAP is BSD-3.
  - Blender is GPL-2.0+ (binary use does not require source disclosure
    unless you modify and distribute Blender itself).
  - User-supplied video is assumed proprietary / all-rights-reserved
    unless the user asserts otherwise.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ToolEntry:
    name: str
    version: Optional[str] = None
    license: str = "unknown"
    license_url: Optional[str] = None
    source_url: Optional[str] = None
    risk_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SourceEntry:
    source_type: str              # user_video | user_images | third_party
    description: str = ""
    license: str = "proprietary"  # assumed unless user asserts otherwise
    attribution: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class LicenseManifest:
    asset_id: str
    sources: List[SourceEntry] = field(default_factory=list)
    tools: List[ToolEntry] = field(default_factory=list)
    output_license: str = "proprietary"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "sources": [s.to_dict() for s in self.sources],
            "tools": [t.to_dict() for t in self.tools],
            "output_license": self.output_license,
            "notes": self.notes,
        }

    def write(self, output_path: "str | Path") -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ──────────────────────── known tool registry ────────────────────────

KNOWN_TOOLS: Dict[str, ToolEntry] = {
    "colmap": ToolEntry(
        name="COLMAP",
        license="BSD-3-Clause",
        license_url="https://github.com/colmap/colmap/blob/main/LICENSE",
        source_url="https://github.com/colmap/colmap",
    ),
    "openmvs": ToolEntry(
        name="OpenMVS",
        license="AGPL-3.0",
        license_url="https://github.com/cdcseacave/openMVS/blob/master/LICENSE",
        source_url="https://github.com/cdcseacave/openMVS",
        risk_note=(
            "OpenMVS is AGPL-3.0. If you distribute a product or service "
            "that incorporates OpenMVS, you may need to provide source code. "
            "Consult your legal team before commercial distribution."
        ),
    ),
    "blender": ToolEntry(
        name="Blender",
        license="GPL-2.0-or-later",
        license_url="https://www.blender.org/about/license/",
        source_url="https://projects.blender.org/blender/blender",
        risk_note=(
            "Blender is GPL. Using Blender as an external tool (subprocess) "
            "does not require you to GPL-license your own code, but "
            "redistribution of modified Blender binaries requires source disclosure."
        ),
    ),
    "gltf_transform": ToolEntry(
        name="glTF-Transform",
        license="MIT",
        license_url="https://github.com/donmccurdy/glTF-Transform/blob/main/LICENSE",
        source_url="https://github.com/donmccurdy/glTF-Transform",
    ),
    "gltf_validator": ToolEntry(
        name="glTF Validator (Khronos)",
        license="Apache-2.0",
        license_url="https://github.com/KhronosGroup/glTF-Validator/blob/main/LICENSE",
        source_url="https://github.com/KhronosGroup/glTF-Validator",
    ),
    "ffmpeg": ToolEntry(
        name="FFmpeg",
        license="LGPL-2.1 / GPL-2.0 (depends on build)",
        license_url="https://www.ffmpeg.org/legal.html",
        source_url="https://ffmpeg.org",
        risk_note="FFmpeg license depends on which codecs are compiled in.",
    ),
}


def build_license_manifest(
    asset_id: str,
    source_video_path: Optional[str] = None,
    active_tools: Optional[List[str]] = None,
    extra_notes: Optional[List[str]] = None,
) -> LicenseManifest:
    """
    Build a LicenseManifest for a reconstruction job.

    active_tools: list of keys from KNOWN_TOOLS; defaults to ["colmap", "openmvs"].
    """
    if active_tools is None:
        active_tools = ["colmap", "openmvs"]

    sources: List[SourceEntry] = []
    if source_video_path:
        sources.append(SourceEntry(
            source_type="user_video",
            description=f"User-supplied capture video: {Path(source_video_path).name}",
            license="proprietary",
        ))
    else:
        sources.append(SourceEntry(
            source_type="user_images",
            description="User-supplied capture images",
            license="proprietary",
        ))

    tools = [KNOWN_TOOLS[k] for k in active_tools if k in KNOWN_TOOLS]
    notes = list(extra_notes or [])
    # Auto-surface AGPL risk when OpenMVS is used
    if "openmvs" in active_tools:
        notes.append("⚠ OpenMVS (AGPL-3.0) was used. Review distribution obligations before publishing.")

    return LicenseManifest(
        asset_id=asset_id,
        sources=sources,
        tools=tools,
        output_license="proprietary",
        notes=notes,
    )
