"""
Command Config — convert a preset dict into typed COLMAP/OpenMVS run params.

Single source of truth for which preset value gets pushed to which CLI flag.
ColmapCommandBuilder + OpenMVSTexturer + future adapters consume this object;
they don't touch preset dicts directly.

Default `baseline_command_config()` matches the current env-driven behavior, so
hardening-disabled paths can call the same code with no behavior change.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional


@dataclass
class ColmapCommandConfig:
    feature_quality: str = "high"          # low | medium | high
    matcher_type: str = "exhaustive"       # exhaustive | sequential
    max_image_size: int = 2000
    mapper_min_num_matches: int = 15
    patchmatch_resolution_level: int = 1   # 0=full, 1=half, 2=quarter

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OpenMVSCommandConfig:
    texture_resolution: int = 4096
    max_threads: int = 0                   # 0 = let OpenMVS decide
    enable_texture_retry: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReconstructionCommandConfig:
    """Top-level container the runner passes to adapters."""
    colmap: ColmapCommandConfig = field(default_factory=ColmapCommandConfig)
    openmvs: OpenMVSCommandConfig = field(default_factory=OpenMVSCommandConfig)
    source_preset_name: str = "baseline"
    rationale: str = "default baseline matching env settings"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applied": True,
            "source_preset_name": self.source_preset_name,
            "rationale": self.rationale,
            "colmap": self.colmap.to_dict(),
            "openmvs": self.openmvs.to_dict(),
        }


def baseline_command_config() -> ReconstructionCommandConfig:
    """
    Mirror of env-default behavior.  Always safe to call; never None.
    """
    return ReconstructionCommandConfig(
        colmap=ColmapCommandConfig(),
        openmvs=OpenMVSCommandConfig(),
        source_preset_name="baseline",
        rationale="default baseline matching env settings",
    )


def from_preset(preset: Optional[Dict[str, Any]]) -> ReconstructionCommandConfig:
    """
    Build a typed command config from a preset dict (output of
    `reconstruction_preset_resolver.resolve_preset()` or
    `fallback_ladder.pick_next_preset().preset_snapshot`).

    Missing keys fall back to baseline values — partial presets won't crash.
    """
    if not preset or not isinstance(preset, dict):
        return baseline_command_config()

    base = baseline_command_config()
    colmap_in = preset.get("colmap")
    openmvs_in = preset.get("openmvs")
    if not isinstance(colmap_in, dict):
        colmap_in = {}
    if not isinstance(openmvs_in, dict):
        openmvs_in = {}

    cfg_colmap = ColmapCommandConfig(
        feature_quality=str(colmap_in.get("feature_quality", base.colmap.feature_quality)),
        matcher_type=str(colmap_in.get("matcher_type", base.colmap.matcher_type)),
        max_image_size=int(colmap_in.get("max_image_size", base.colmap.max_image_size)),
        mapper_min_num_matches=int(colmap_in.get("mapper_min_num_matches", base.colmap.mapper_min_num_matches)),
        patchmatch_resolution_level=int(
            colmap_in.get("patchmatch_resolution_level", base.colmap.patchmatch_resolution_level)
        ),
    )
    cfg_openmvs = OpenMVSCommandConfig(
        texture_resolution=int(openmvs_in.get("texture_resolution", base.openmvs.texture_resolution)),
        max_threads=int(openmvs_in.get("max_threads", base.openmvs.max_threads)),
        enable_texture_retry=bool(openmvs_in.get("enable_texture_retry", base.openmvs.enable_texture_retry)),
    )

    return ReconstructionCommandConfig(
        colmap=cfg_colmap,
        openmvs=cfg_openmvs,
        source_preset_name=str(preset.get("name", "unknown")),
        rationale=str(preset.get("rationale", "")),
    )
