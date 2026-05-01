"""Sprint 1 — scorecard schema + grade aggregation tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh

from modules.qa_validation.scorecard import (
    SCHEMA_VERSION,
    build_scorecard,
    write_scorecard,
)


def test_scorecard_skeleton_with_no_inputs(tmp_path):
    sc = build_scorecard(job_id="job_x", job_dir=tmp_path)
    assert sc["schema_version"] == SCHEMA_VERSION
    assert sc["job_id"] == "job_x"
    assert "generated_at" in sc
    for key in ("coverage", "geometry", "texture", "reconstruction",
                "capture_profile", "color_profile", "overall"):
        assert key in sc
    assert sc["overall"]["grade"] == "F"
    assert sc["overall"]["production_ready"] is False


def test_scorecard_with_clean_box_grades_well(tmp_path):
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    sc = build_scorecard(job_id="job_box", job_dir=tmp_path, mesh=mesh)
    assert sc["geometry"]["grade"] == "A"
    # No texture/cameras → overall caps out at C/F due to texture absence
    assert sc["overall"]["grade"] in ("C", "F")


def test_scorecard_writes_quality_report_json(tmp_path):
    sc = build_scorecard(job_id="job_y", job_dir=tmp_path)
    out_path = write_scorecard(tmp_path, sc)
    assert out_path.name == "quality_report.json"
    assert out_path.exists()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["job_id"] == "job_y"


def test_scorecard_picks_up_extraction_manifest(tmp_path):
    # Drop a fake extraction manifest with capture/color profile hints
    extraction = {
        "capture_profile": {
            "size_class": "large",
            "scene_type": "freestanding",
            "material_hint": "metallic",
            "preset_key": "large__freestanding",
        },
        "color_profile": {
            "category": "vibrant",
            "product_rgb": [200, 40, 40],
            "background_rgb": [230, 230, 230],
        },
    }
    (tmp_path / "extraction_manifest.json").write_text(
        json.dumps(extraction), encoding="utf-8"
    )
    sc = build_scorecard(job_id="job_z", job_dir=tmp_path)
    assert sc["capture_profile"]["size_class"] == "large"
    assert sc["color_profile"]["category"] == "vibrant"


def test_scorecard_blockers_listed(tmp_path):
    sc = build_scorecard(job_id="job_b", job_dir=tmp_path)  # empty inputs
    blockers = sc["overall"]["blockers"]
    # No mesh → reconstruction blocker
    assert any("reconstruction" in b.lower() or "mesh" in b.lower() or "geometry" in b.lower()
               for b in blockers)


def test_scorecard_does_not_raise_on_corrupt_manifest(tmp_path):
    (tmp_path / "extraction_manifest.json").write_text("{ this is not valid json", encoding="utf-8")
    sc = build_scorecard(job_id="job_c", job_dir=tmp_path)
    assert sc["job_id"] == "job_c"  # ran to completion despite broken manifest
