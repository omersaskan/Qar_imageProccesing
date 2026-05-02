"""Sprint 9 — Sprint 6 Blender headless cleanup pipeline tests."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ─────────────────────────── mesh_normalization ───────────────────────────

from modules.asset_cleanup.mesh_normalization import NormalizationConfig


def test_normalization_config_defaults():
    cfg = NormalizationConfig()
    assert cfg.align_to_origin is True
    assert cfg.align_ground_to_z_zero is True
    assert cfg.apply_scale is True
    assert cfg.decimate_enabled is False
    assert cfg.decimate_ratio == 0.5
    assert cfg.forward_axis == "-Z"
    assert cfg.up_axis == "Y"


def test_normalization_config_to_dict():
    cfg = NormalizationConfig(decimate_enabled=True, decimate_ratio=0.3)
    d = cfg.to_dict()
    assert d["decimate_enabled"] is True
    assert d["decimate_ratio"] == pytest.approx(0.3)


def test_normalization_config_from_dict_partial():
    cfg = NormalizationConfig.from_dict({"decimate_enabled": True, "decimate_ratio": 0.25})
    assert cfg.decimate_enabled is True
    assert cfg.decimate_ratio == pytest.approx(0.25)
    # Unset fields keep defaults
    assert cfg.align_to_origin is True


def test_normalization_config_from_dict_ignores_unknown_keys():
    cfg = NormalizationConfig.from_dict({"unknown_key": "value"})
    assert cfg.align_to_origin is True  # default unchanged


def test_normalization_config_round_trip():
    cfg = NormalizationConfig(decimate_enabled=True, decimate_ratio=0.7, image_format="JPEG")
    restored = NormalizationConfig.from_dict(cfg.to_dict())
    assert restored.decimate_enabled == cfg.decimate_enabled
    assert restored.decimate_ratio == pytest.approx(cfg.decimate_ratio)
    assert restored.image_format == "JPEG"


# ─────────────────────────── blender_script_generator ───────────────────────────

from modules.asset_cleanup.blender_script_generator import generate_cleanup_script


def test_script_contains_input_path():
    cfg = NormalizationConfig()
    script = generate_cleanup_script("/input/mesh.obj", "/output/mesh.glb", cfg)
    assert "/input/mesh.obj" in script


def test_script_contains_output_path():
    cfg = NormalizationConfig()
    script = generate_cleanup_script("/input/mesh.obj", "/output/mesh.glb", cfg)
    assert "/output/mesh.glb" in script


def test_script_decimate_disabled_by_default():
    cfg = NormalizationConfig(decimate_enabled=False)
    script = generate_cleanup_script("/input/mesh.obj", "/output/mesh.glb", cfg)
    assert "DECIMATE = False" in script


def test_script_decimate_enabled_reflected():
    cfg = NormalizationConfig(decimate_enabled=True, decimate_ratio=0.4)
    script = generate_cleanup_script("/input/mesh.obj", "/output/mesh.glb", cfg)
    assert "DECIMATE = True" in script
    assert "0.4" in script


def test_script_align_to_origin_reflected():
    cfg = NormalizationConfig(align_to_origin=False)
    script = generate_cleanup_script("/input/mesh.obj", "/output/mesh.glb", cfg)
    assert "ALIGN_ORIGIN = False" in script


def test_script_is_valid_python_syntax():
    import ast
    cfg = NormalizationConfig()
    script = generate_cleanup_script("/input/mesh.obj", "/out/mesh.glb", cfg)
    ast.parse(script)  # raises SyntaxError if invalid


def test_script_sys_exit_present():
    cfg = NormalizationConfig()
    script = generate_cleanup_script("/a.obj", "/b.glb", cfg)
    assert "sys.exit(0)" in script
    assert "sys.exit(1)" in script


# ─────────────────────────── blender_headless_worker ───────────────────────────

from modules.asset_cleanup.blender_headless_worker import (
    BlenderWorkerResult,
    run_blender_cleanup,
    _find_blender,
    _blender_version,
)


def test_blender_result_to_dict():
    r = BlenderWorkerResult(status="ok", output_glb="/out/mesh.glb", elapsed_seconds=12.5)
    d = r.to_dict()
    assert d["status"] == "ok"
    assert d["output_glb"] == "/out/mesh.glb"
    assert d["elapsed_seconds"] == pytest.approx(12.5)


def test_run_blender_cleanup_unavailable_when_not_on_path(monkeypatch, tmp_path):
    monkeypatch.setenv("BLENDER_BIN", "")
    monkeypatch.setattr(
        "modules.asset_cleanup.blender_headless_worker.shutil.which",
        lambda _: None,
    )
    result = run_blender_cleanup("/input.obj", str(tmp_path / "out.glb"))
    assert result.status == "unavailable"
    assert "blender" in result.reason.lower()


def test_run_blender_cleanup_missing_input(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "modules.asset_cleanup.blender_headless_worker._find_blender",
        lambda: "/fake/blender",
    )
    monkeypatch.setattr(
        "modules.asset_cleanup.blender_headless_worker._blender_version",
        lambda b: "Blender 4.0.0",
    )
    result = run_blender_cleanup(
        str(tmp_path / "nonexistent.obj"),
        str(tmp_path / "out.glb"),
    )
    assert result.status == "failed"
    assert "not found" in result.reason


def test_run_blender_cleanup_success(monkeypatch, tmp_path):
    input_obj = tmp_path / "mesh.obj"
    input_obj.write_text("# obj\n")
    output_glb = tmp_path / "out" / "clean.glb"

    monkeypatch.setattr(
        "modules.asset_cleanup.blender_headless_worker._find_blender",
        lambda: "/fake/blender",
    )
    monkeypatch.setattr(
        "modules.asset_cleanup.blender_headless_worker._blender_version",
        lambda b: "Blender 4.0.0",
    )

    class FakeCompleted:
        returncode = 0
        stdout = "Blender done\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        output_glb.parent.mkdir(parents=True, exist_ok=True)
        output_glb.write_bytes(b"FAKE_GLB")
        return FakeCompleted()

    monkeypatch.setattr("subprocess.run", fake_run)
    result = run_blender_cleanup(str(input_obj), str(output_glb))
    assert result.status == "ok"
    assert result.output_glb == str(output_glb)
    assert "Blender 4.0.0" == result.blender_version


def test_run_blender_cleanup_nonzero_exit(monkeypatch, tmp_path):
    input_obj = tmp_path / "mesh.obj"
    input_obj.write_text("# obj\n")

    monkeypatch.setattr(
        "modules.asset_cleanup.blender_headless_worker._find_blender",
        lambda: "/fake/blender",
    )
    monkeypatch.setattr(
        "modules.asset_cleanup.blender_headless_worker._blender_version",
        lambda b: None,
    )

    class FakeCompleted:
        returncode = 1
        stdout = "error in script\n"
        stderr = "Traceback...\n"

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: FakeCompleted())
    result = run_blender_cleanup(str(input_obj), str(tmp_path / "out.glb"))
    assert result.status == "failed"
    assert "code 1" in result.reason


def test_run_blender_cleanup_timeout(monkeypatch, tmp_path):
    import subprocess
    input_obj = tmp_path / "mesh.obj"
    input_obj.write_text("# obj\n")

    monkeypatch.setattr(
        "modules.asset_cleanup.blender_headless_worker._find_blender",
        lambda: "/fake/blender",
    )
    monkeypatch.setattr(
        "modules.asset_cleanup.blender_headless_worker._blender_version",
        lambda b: None,
    )

    def raise_timeout(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 30)

    monkeypatch.setattr("subprocess.run", raise_timeout)
    result = run_blender_cleanup(str(input_obj), str(tmp_path / "out.glb"), timeout_seconds=30)
    assert result.status == "failed"
    assert "timed out" in result.reason


# ─────────────────────────── glb_export_manifest ───────────────────────────

from modules.export_pipeline.glb_export_manifest import (
    build_blender_cleanup_block,
    write_blender_cleanup_sidecar,
)


def test_build_blender_cleanup_block_ok():
    r = BlenderWorkerResult(status="ok", output_glb="/out/mesh.glb", blender_version="Blender 4.0")
    block = build_blender_cleanup_block(r, original_mesh_path="/input/mesh.obj")
    assert block["status"] == "ok"
    assert block["output_glb"] == "/out/mesh.glb"
    assert block["original_mesh_path"] == "/input/mesh.obj"


def test_build_blender_cleanup_block_unavailable():
    r = BlenderWorkerResult(status="unavailable", reason="not installed")
    block = build_blender_cleanup_block(r)
    assert block["status"] == "unavailable"
    assert "not installed" in block["reason"]


def test_write_blender_cleanup_sidecar(tmp_path):
    block = {"status": "ok", "output_glb": "/out/mesh.glb"}
    write_blender_cleanup_sidecar(tmp_path, block)
    written = json.loads((tmp_path / "blender_cleanup.json").read_text(encoding="utf-8"))
    assert written["status"] == "ok"
