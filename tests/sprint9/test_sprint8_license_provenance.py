"""Sprint 9 — Sprint 8 license manifest + asset provenance tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ─────────────────────────── license_manifest ───────────────────────────

from modules.asset_registry.license_manifest import (
    LicenseManifest,
    ToolEntry,
    SourceEntry,
    KNOWN_TOOLS,
    build_license_manifest,
)


def test_known_tools_contains_required_tools():
    for key in ("colmap", "openmvs", "blender", "gltf_transform", "gltf_validator", "ffmpeg"):
        assert key in KNOWN_TOOLS, f"Missing tool: {key}"


def test_openmvs_has_agpl_license():
    assert "AGPL" in KNOWN_TOOLS["openmvs"].license


def test_openmvs_has_risk_note():
    assert KNOWN_TOOLS["openmvs"].risk_note is not None
    assert len(KNOWN_TOOLS["openmvs"].risk_note) > 10


def test_colmap_is_bsd():
    assert "BSD" in KNOWN_TOOLS["colmap"].license


def test_blender_is_gpl():
    assert "GPL" in KNOWN_TOOLS["blender"].license


def test_tool_entry_to_dict_excludes_none():
    t = ToolEntry(name="COLMAP", license="BSD-3-Clause", version=None)
    d = t.to_dict()
    assert "version" not in d
    assert d["name"] == "COLMAP"


def test_source_entry_defaults_to_proprietary():
    s = SourceEntry(source_type="user_video", description="test.mp4")
    assert s.license == "proprietary"


def test_license_manifest_to_dict():
    lm = LicenseManifest(
        asset_id="asset-123",
        sources=[SourceEntry(source_type="user_images")],
        tools=[KNOWN_TOOLS["colmap"]],
    )
    d = lm.to_dict()
    assert d["asset_id"] == "asset-123"
    assert len(d["sources"]) == 1
    assert len(d["tools"]) == 1
    json.dumps(d)  # serialisable


def test_build_license_manifest_default_tools():
    lm = build_license_manifest(asset_id="a1", source_video_path="/videos/capture.mp4")
    d = lm.to_dict()
    tool_names = [t["name"] for t in d["tools"]]
    assert "COLMAP" in tool_names
    assert "OpenMVS" in tool_names


def test_build_license_manifest_agpl_note_included_when_openmvs():
    lm = build_license_manifest(asset_id="a1", active_tools=["colmap", "openmvs"])
    assert any("AGPL" in n for n in lm.notes)


def test_build_license_manifest_no_agpl_note_without_openmvs():
    lm = build_license_manifest(asset_id="a1", active_tools=["colmap"])
    assert not any("AGPL" in n for n in lm.notes)


def test_build_license_manifest_with_blender():
    lm = build_license_manifest(asset_id="a1", active_tools=["colmap", "blender"])
    tool_names = [t["name"] for t in lm.to_dict()["tools"]]
    assert "Blender" in tool_names


def test_license_manifest_write(tmp_path):
    lm = build_license_manifest(asset_id="a2", active_tools=["colmap", "openmvs"])
    out = tmp_path / "license_manifest.json"
    lm.write(out)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["asset_id"] == "a2"
    assert any("AGPL" in n for n in data["notes"])


def test_license_manifest_write_creates_parent_dirs(tmp_path):
    lm = build_license_manifest(asset_id="a3")
    deep = tmp_path / "a" / "b" / "license_manifest.json"
    lm.write(deep)
    assert deep.exists()


def test_license_manifest_output_license_default_proprietary():
    lm = build_license_manifest(asset_id="a4")
    assert lm.output_license == "proprietary"


# ─────────────────────────── asset_provenance ───────────────────────────

from modules.asset_registry.asset_provenance import (
    AssetProvenance,
    ProvenanceStep,
    provenance_from_manifest,
)


def test_provenance_step_to_dict_excludes_empty():
    step = ProvenanceStep(step="reconstruction", status="ok")
    d = step.to_dict()
    assert d["step"] == "reconstruction"
    assert "input_paths" not in d  # empty list excluded
    assert "params" not in d       # None excluded


def test_asset_provenance_add_step():
    prov = AssetProvenance(asset_id="a", job_id="j", capture_session_id="s")
    prov.add_step(ProvenanceStep(step="reconstruction", status="ok"))
    assert len(prov.steps) == 1


def test_asset_provenance_to_dict():
    prov = AssetProvenance(asset_id="a", job_id="j", capture_session_id="s")
    prov.add_step(ProvenanceStep(step="reconstruction", status="ok", tool="colmap_openmvs"))
    d = prov.to_dict()
    assert d["asset_id"] == "a"
    assert len(d["steps"]) == 1
    json.dumps(d)


def test_asset_provenance_write(tmp_path):
    prov = AssetProvenance(asset_id="a", job_id="j", capture_session_id="s")
    prov.add_step(ProvenanceStep(step="reconstruction", status="ok"))
    out = tmp_path / "asset_provenance.json"
    prov.write(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["asset_id"] == "a"
    assert len(data["steps"]) == 1


def test_provenance_from_manifest_basic():
    manifest = {
        "engine_type": "colmap_openmvs",
        "mesh_path": "/out/mesh.obj",
        "texture_path": "/out/texture.png",
        "texturing_engine": "openmvs_texturemesh",
    }
    prov = provenance_from_manifest("a", "j", "s", manifest)
    step_names = [s.step for s in prov.steps]
    assert "reconstruction" in step_names
    assert "texturing" in step_names


def test_provenance_from_manifest_with_blender_cleanup():
    manifest = {
        "engine_type": "colmap_openmvs",
        "mesh_path": "/out/mesh.obj",
        "blender_cleanup": {
            "status": "ok",
            "output_glb": "/out/clean.glb",
            "blender_version": "Blender 4.0",
            "original_mesh_path": "/out/mesh.obj",
        },
    }
    prov = provenance_from_manifest("a", "j", "s", manifest)
    step_names = [s.step for s in prov.steps]
    assert "blender_cleanup" in step_names
    bc = next(s for s in prov.steps if s.step == "blender_cleanup")
    assert bc.tool_version == "Blender 4.0"


def test_provenance_from_manifest_with_gltf_steps():
    manifest = {
        "engine_type": "colmap_openmvs",
        "mesh_path": "/out/mesh.obj",
        "gltf_optimization": {"status": "ok", "output_glb": "/out/opt.glb", "cli_version": "4.0"},
        "gltf_validation": {"status": "warning", "error_count": 0, "warning_count": 2},
    }
    prov = provenance_from_manifest("a", "j", "s", manifest)
    step_names = [s.step for s in prov.steps]
    assert "gltf_optimization" in step_names
    assert "gltf_validation" in step_names


def test_provenance_from_manifest_missing_mesh_still_records():
    manifest = {"engine_type": "colmap_openmvs"}
    prov = provenance_from_manifest("a", "j", "s", manifest)
    assert any(s.step == "reconstruction" for s in prov.steps)
    recon = next(s for s in prov.steps if s.step == "reconstruction")
    assert recon.status == "failed"


def test_provenance_created_at_is_utc_iso():
    prov = AssetProvenance(asset_id="a", job_id="j", capture_session_id="s")
    from datetime import datetime
    dt = datetime.fromisoformat(prov.created_at.replace("Z", "+00:00"))
    assert dt.tzinfo is not None


def test_provenance_pipeline_version():
    prov = AssetProvenance(asset_id="a", job_id="j", capture_session_id="s")
    assert prov.pipeline_version == "sprint-8"
