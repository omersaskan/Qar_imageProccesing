"""Sprint 9 — Sprint 7 glTF-Transform + Validator + AR gate tests."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ─────────────────────────── gltf_transform_optimizer ───────────────────────────

from modules.export_pipeline.gltf_transform_optimizer import (
    GltfTransformConfig,
    GltfTransformResult,
    optimize_glb,
    _find_gltf_transform,
)


def test_gltf_transform_config_defaults():
    cfg = GltfTransformConfig()
    assert cfg.prune is True
    assert cfg.dedup is True
    assert cfg.flatten is False
    assert cfg.draco_compression is False
    assert cfg.resize_textures is False


def test_gltf_transform_result_to_dict():
    r = GltfTransformResult(status="ok", output_glb="/out/opt.glb", elapsed_seconds=5.0)
    d = r.to_dict()
    assert d["status"] == "ok"
    json.dumps(d)  # serialisable


def test_optimize_glb_unavailable_when_cli_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("GLTF_TRANSFORM_BIN", "")
    monkeypatch.setattr(
        "modules.export_pipeline.gltf_transform_optimizer.shutil.which",
        lambda _: None,
    )
    result = optimize_glb(str(tmp_path / "input.glb"), str(tmp_path / "out.glb"))
    assert result.status == "unavailable"
    assert "gltf-transform" in result.reason.lower()


def test_optimize_glb_missing_input(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "modules.export_pipeline.gltf_transform_optimizer._find_gltf_transform",
        lambda: "/fake/gltf-transform",
    )
    result = optimize_glb(str(tmp_path / "noexist.glb"), str(tmp_path / "out.glb"))
    assert result.status == "failed"
    assert "not found" in result.reason


def test_optimize_glb_success(monkeypatch, tmp_path):
    input_glb = tmp_path / "input.glb"
    input_glb.write_bytes(b"FAKE")
    output_glb = tmp_path / "out.glb"

    monkeypatch.setattr(
        "modules.export_pipeline.gltf_transform_optimizer._find_gltf_transform",
        lambda: "/fake/gltf-transform",
    )
    monkeypatch.setattr(
        "modules.export_pipeline.gltf_transform_optimizer._cli_version",
        lambda b: "4.0.0",
    )

    class FakeRun:
        returncode = 0
        stdout = "Done"
        stderr = ""

    def fake_subprocess(cmd, **kw):
        output_glb.write_bytes(b"OPT_GLB")
        return FakeRun()

    monkeypatch.setattr("subprocess.run", fake_subprocess)
    result = optimize_glb(str(input_glb), str(output_glb))
    assert result.status == "ok"
    assert result.output_glb == str(output_glb)
    assert result.cli_version == "4.0.0"
    assert "optimize" in result.operations_run


def test_optimize_glb_nonzero_exit(monkeypatch, tmp_path):
    input_glb = tmp_path / "input.glb"
    input_glb.write_bytes(b"FAKE")

    monkeypatch.setattr(
        "modules.export_pipeline.gltf_transform_optimizer._find_gltf_transform",
        lambda: "/fake/gltf-transform",
    )
    monkeypatch.setattr(
        "modules.export_pipeline.gltf_transform_optimizer._cli_version",
        lambda b: None,
    )

    class FakeRun:
        returncode = 2
        stdout = ""
        stderr = "error"

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: FakeRun())
    result = optimize_glb(str(input_glb), str(tmp_path / "out.glb"))
    assert result.status == "failed"
    assert "2" in result.reason


def test_optimize_glb_timeout(monkeypatch, tmp_path):
    input_glb = tmp_path / "input.glb"
    input_glb.write_bytes(b"FAKE")

    monkeypatch.setattr(
        "modules.export_pipeline.gltf_transform_optimizer._find_gltf_transform",
        lambda: "/fake/gltf-transform",
    )
    monkeypatch.setattr(
        "modules.export_pipeline.gltf_transform_optimizer._cli_version",
        lambda b: None,
    )

    def raise_timeout(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 300)

    monkeypatch.setattr("subprocess.run", raise_timeout)
    result = optimize_glb(str(input_glb), str(tmp_path / "out.glb"))
    assert result.status == "failed"
    assert "timed out" in result.reason


# ─────────────────────────── gltf_validator ───────────────────────────

from modules.qa_validation.gltf_validator import (
    GltfValidationReport,
    validate_glb,
    _find_validator,
)


def test_validate_glb_unavailable_when_cli_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("GLTF_VALIDATOR_BIN", "")
    monkeypatch.setattr(
        "modules.qa_validation.gltf_validator.shutil.which",
        lambda _: None,
    )
    result = validate_glb(str(tmp_path / "mesh.glb"))
    assert result.status == "unavailable"


def test_validate_glb_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "modules.qa_validation.gltf_validator._find_validator",
        lambda: "/fake/gltf_validator",
    )
    result = validate_glb(str(tmp_path / "noexist.glb"))
    assert result.status == "failed"
    assert "not found" in result.reason


def _validator_json(num_errors=0, num_warnings=0):
    return json.dumps({
        "issues": {
            "numErrors": num_errors,
            "numWarnings": num_warnings,
            "numInfos": 0,
            "messages": [],
        },
        "info": {"version": "2.0", "generator": "TestEngine"},
    })


def test_validate_glb_ok(monkeypatch, tmp_path):
    glb = tmp_path / "mesh.glb"
    glb.write_bytes(b"FAKE")

    monkeypatch.setattr(
        "modules.qa_validation.gltf_validator._find_validator",
        lambda: "/fake/gltf_validator",
    )

    class FakeRun:
        returncode = 0
        stdout = _validator_json(0, 0)
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: FakeRun())
    result = validate_glb(str(glb))
    assert result.status == "ok"
    assert result.error_count == 0


def test_validate_glb_with_errors(monkeypatch, tmp_path):
    glb = tmp_path / "mesh.glb"
    glb.write_bytes(b"FAKE")

    monkeypatch.setattr(
        "modules.qa_validation.gltf_validator._find_validator",
        lambda: "/fake/gltf_validator",
    )

    class FakeRun:
        returncode = 1
        stdout = _validator_json(3, 1)
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: FakeRun())
    result = validate_glb(str(glb))
    assert result.status == "error"
    assert result.error_count == 3
    assert result.warning_count == 1


def test_validate_glb_with_warnings_only(monkeypatch, tmp_path):
    glb = tmp_path / "mesh.glb"
    glb.write_bytes(b"FAKE")

    monkeypatch.setattr(
        "modules.qa_validation.gltf_validator._find_validator",
        lambda: "/fake/gltf_validator",
    )

    class FakeRun:
        returncode = 0
        stdout = _validator_json(0, 2)
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: FakeRun())
    result = validate_glb(str(glb))
    assert result.status == "warning"
    assert result.warning_count == 2


def test_validate_glb_timeout(monkeypatch, tmp_path):
    glb = tmp_path / "mesh.glb"
    glb.write_bytes(b"FAKE")

    monkeypatch.setattr(
        "modules.qa_validation.gltf_validator._find_validator",
        lambda: "/fake/gltf_validator",
    )

    def raise_timeout(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 60)

    monkeypatch.setattr("subprocess.run", raise_timeout)
    result = validate_glb(str(glb))
    assert result.status == "failed"
    assert "timed out" in result.reason


def test_validate_glb_report_to_dict_serialisable(monkeypatch, tmp_path):
    glb = tmp_path / "mesh.glb"
    glb.write_bytes(b"FAKE")

    monkeypatch.setattr(
        "modules.qa_validation.gltf_validator._find_validator",
        lambda: "/fake/gltf_validator",
    )

    class FakeRun:
        returncode = 0
        stdout = _validator_json(0, 0)
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: FakeRun())
    result = validate_glb(str(glb))
    json.dumps(result.to_dict())


# ─────────────────────────── ar_asset_gate ───────────────────────────

from modules.qa_validation.ar_asset_gate import evaluate_ar_gate, ArAssetGateResult
from modules.export_pipeline.gltf_transform_optimizer import GltfTransformResult
from modules.qa_validation.gltf_validator import GltfValidationReport


def test_ar_gate_pass_on_both_ok():
    opt = GltfTransformResult(status="ok")
    val = GltfValidationReport(status="ok")
    result = evaluate_ar_gate(opt, val)
    assert result.verdict == "pass"


def test_ar_gate_reject_on_validator_error():
    opt = GltfTransformResult(status="ok")
    val = GltfValidationReport(status="error", error_count=2)
    result = evaluate_ar_gate(opt, val)
    assert result.verdict == "reject"
    assert any("error" in r.lower() for r in result.reasons)


def test_ar_gate_review_on_validator_warning():
    opt = GltfTransformResult(status="ok")
    val = GltfValidationReport(status="warning", warning_count=3)
    result = evaluate_ar_gate(opt, val)
    assert result.verdict == "review"


def test_ar_gate_review_on_optimizer_failed():
    opt = GltfTransformResult(status="failed", reason="crash")
    val = GltfValidationReport(status="ok")
    result = evaluate_ar_gate(opt, val)
    assert result.verdict == "review"


def test_ar_gate_pass_on_optimizer_unavailable():
    opt = GltfTransformResult(status="unavailable")
    val = GltfValidationReport(status="ok")
    result = evaluate_ar_gate(opt, val)
    assert result.verdict == "pass"


def test_ar_gate_reject_overrides_review():
    """Validator error takes precedence over optimizer review."""
    opt = GltfTransformResult(status="failed")
    val = GltfValidationReport(status="error", error_count=1)
    result = evaluate_ar_gate(opt, val)
    assert result.verdict == "reject"


def test_ar_gate_none_inputs_pass():
    result = evaluate_ar_gate(None, None)
    assert result.verdict == "pass"


def test_ar_gate_result_to_dict_serialisable():
    result = evaluate_ar_gate(
        GltfTransformResult(status="ok"),
        GltfValidationReport(status="warning", warning_count=1),
    )
    d = result.to_dict()
    json.dumps(d)
