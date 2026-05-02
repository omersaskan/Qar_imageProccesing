"""
Asset provenance — Sprint 8.

Records the full tool-chain used to produce a 3D asset along with
timestamps, versions, and pipeline step outcomes.  Written as
asset_provenance.json next to the final asset.

Provenance is append-only: each pipeline step adds a record.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ProvenanceStep:
    step: str                          # e.g. "colmap_sfm", "openmvs_texture", "blender_cleanup"
    status: str                        # ok | failed | skipped | unavailable
    tool: Optional[str] = None
    tool_version: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    input_paths: List[str] = field(default_factory=list)
    output_paths: List[str] = field(default_factory=list)
    params: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v != []}


@dataclass
class AssetProvenance:
    asset_id: str
    job_id: str
    capture_session_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    steps: List[ProvenanceStep] = field(default_factory=list)
    pipeline_version: str = "sprint-8"

    def add_step(self, step: ProvenanceStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "job_id": self.job_id,
            "capture_session_id": self.capture_session_id,
            "created_at": self.created_at,
            "pipeline_version": self.pipeline_version,
            "steps": [s.to_dict() for s in self.steps],
        }

    def write(self, output_path: "str | Path") -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def provenance_from_manifest(
    asset_id: str,
    job_id: str,
    capture_session_id: str,
    manifest_data: Dict[str, Any],
) -> AssetProvenance:
    """
    Build an AssetProvenance from a completed manifest dict.

    Reads keys added by each sprint to reconstruct the pipeline trace.
    Missing keys → step recorded as skipped.
    """
    prov = AssetProvenance(
        asset_id=asset_id,
        job_id=job_id,
        capture_session_id=capture_session_id,
    )

    # Reconstruction (Sprint 1/4.6)
    engine = manifest_data.get("engine_type", "unknown")
    recon_status = "ok" if manifest_data.get("mesh_path") else "failed"
    prov.add_step(ProvenanceStep(
        step="reconstruction",
        status=recon_status,
        tool=engine,
        output_paths=[manifest_data["mesh_path"]] if manifest_data.get("mesh_path") else [],
    ))

    # Texturing
    if manifest_data.get("texture_path"):
        prov.add_step(ProvenanceStep(
            step="texturing",
            status="ok",
            tool=manifest_data.get("texturing_engine", "openmvs_texturemesh"),
            output_paths=[manifest_data["texture_path"]],
        ))

    # Sprint 6: Blender cleanup
    blender = manifest_data.get("blender_cleanup")
    if blender:
        prov.add_step(ProvenanceStep(
            step="blender_cleanup",
            status=blender.get("status", "unknown"),
            tool="blender",
            tool_version=blender.get("blender_version"),
            input_paths=[blender.get("original_mesh_path")] if blender.get("original_mesh_path") else [],
            output_paths=[blender.get("output_glb")] if blender.get("output_glb") else [],
        ))

    # Sprint 7: glTF-Transform
    opt = manifest_data.get("gltf_optimization")
    if opt:
        prov.add_step(ProvenanceStep(
            step="gltf_optimization",
            status=opt.get("status", "unknown"),
            tool="gltf-transform",
            tool_version=opt.get("cli_version"),
            output_paths=[opt.get("output_glb")] if opt.get("output_glb") else [],
        ))

    # Sprint 7: Validation
    val = manifest_data.get("gltf_validation")
    if val:
        prov.add_step(ProvenanceStep(
            step="gltf_validation",
            status=val.get("status", "unknown"),
            tool="gltf_validator",
            notes=(
                f"errors={val.get('error_count',0)} "
                f"warnings={val.get('warning_count',0)}"
            ),
        ))

    return prov
