"""Unit tests for Phase 3A benchmark runner."""
import os
import json
import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Ensure repo root on path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.run_ai3d_benchmark import get_mesh_stats, _is_successful, _make_bench_id, run_benchmark


# ─── get_mesh_stats ───────────────────────────────────────────────────────────

def test_get_mesh_stats_missing_file():
    stats = get_mesh_stats("non_existent.glb")
    assert stats["mesh_stats_available"] is False
    assert stats["vertex_count"] == 0


def test_get_mesh_stats_mocked(tmp_path):
    import trimesh
    dummy = tmp_path / "test.glb"
    dummy.touch()

    mock_mesh = MagicMock()
    mock_mesh.vertices = [1, 2, 3]
    mock_mesh.faces = [1, 2]
    mock_scene = MagicMock(spec=trimesh.Scene)
    mock_scene.geometry = {"mesh1": mock_mesh}

    with patch("trimesh.load", return_value=mock_scene):
        stats = get_mesh_stats(str(dummy))

    assert stats["mesh_stats_available"] is True
    assert stats["geometry_count"] == 1
    assert stats["vertex_count"] == 3
    assert stats["face_count"] == 2


# ─── _is_successful ───────────────────────────────────────────────────────────

def test_is_successful_ok(tmp_path):
    glb = tmp_path / "out.glb"
    glb.touch()
    row = {"status": "ok", "provider_status": "ok", "output_glb_path": str(glb)}
    assert _is_successful(row) is True


def test_is_successful_review(tmp_path):
    glb = tmp_path / "out.glb"
    glb.touch()
    row = {"status": "review", "provider_status": "ok", "output_glb_path": str(glb)}
    assert _is_successful(row) is True


def test_is_successful_unavailable():
    row = {"status": "unavailable", "provider_status": "unavailable", "output_glb_path": None}
    assert _is_successful(row) is False


def test_is_successful_missing_glb(tmp_path):
    row = {"status": "ok", "provider_status": "ok", "output_glb_path": str(tmp_path / "ghost.glb")}
    assert _is_successful(row) is False


# ─── _make_bench_id ───────────────────────────────────────────────────────────

def test_make_bench_id_unique():
    id1 = _make_bench_id("img", "high", True)
    id2 = _make_bench_id("img", "high", True)
    assert id1 != id2  # UUID suffix ensures uniqueness


def test_make_bench_id_format():
    bid = _make_bench_id("my_image", "balanced", False)
    assert "my_image" in bid
    assert "balanced" in bid
    assert "bgoff" in bid


# ─── run_benchmark — mocked execution ────────────────────────────────────────

def _mock_manifest(tmp_path):
    glb = tmp_path / "output.glb"
    glb.write_bytes(b"x" * 1024)
    return {
        "session_id": "test_sess",
        "status": "ok",
        "provider_status": "ok",
        "duration_sec": 15.0,
        "output_glb_path": str(glb),
        "output_size_bytes": 1024,
        "peak_mem_mb": 500.0,
        "worker_metadata": {"device": "cuda"},
        "preprocessing": {
            "background_removed": True,
            "mask_source": "rembg",
            "foreground_ratio_estimate": 0.35,
        },
        "candidate_ranking": [{"score": 95.0}],
        "resolved_quality": {"input_size": 1024},
        "input_mode": "single_image",
        "candidate_count": 1,
        "selected_candidate_id": "cand_001",
        "prepared_image_path": "prep.png",
        "warnings": [],
        "errors": [],
    }


def test_benchmark_runner_writes_outputs(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "test.png").touch()
    output_dir = tmp_path / "outputs"

    manifest = _mock_manifest(tmp_path)

    with patch("scripts.run_ai3d_benchmark.generate_ai_3d", return_value=manifest), \
         patch("scripts.run_ai3d_benchmark.get_mesh_stats", return_value={"vertex_count": 0, "face_count": 0, "mesh_stats_available": False, "geometry_count": 0}), \
         patch("scripts.run_ai3d_benchmark._check_sf3d_enabled", return_value=(True, "ok")), \
         patch("scripts.run_ai3d_benchmark._git_sha", return_value="abc1234567"), \
         patch("sys.argv", ["s", "--input-dir", str(input_dir), "--output-dir", str(output_dir), "--modes", "high", "--bg-modes", "off"]):
        run_benchmark()

    assert (output_dir / "results.json").exists()
    assert (output_dir / "results.csv").exists()
    md = (output_dir / "AI_3D_PHASE3A_SF3D_BENCHMARK_REPORT.md")
    assert md.exists()

    content = md.read_text(encoding="utf-8")
    assert "\u2014" in content          # em dash
    assert "abc1234567" in content      # commit SHA
    assert "DRY RUN" not in content     # not a dry run


def test_benchmark_runner_success_count(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "test.png").touch()
    output_dir = tmp_path / "outputs"

    manifest = _mock_manifest(tmp_path)

    with patch("scripts.run_ai3d_benchmark.generate_ai_3d", return_value=manifest), \
         patch("scripts.run_ai3d_benchmark.get_mesh_stats", return_value={"vertex_count": 0, "face_count": 0, "mesh_stats_available": False, "geometry_count": 0}), \
         patch("scripts.run_ai3d_benchmark._check_sf3d_enabled", return_value=(True, "ok")), \
         patch("scripts.run_ai3d_benchmark._git_sha", return_value="abc1234"), \
         patch("sys.argv", ["s", "--input-dir", str(input_dir), "--output-dir", str(output_dir), "--modes", "high", "--bg-modes", "off"]):
        run_benchmark()

    with open(output_dir / "results.json") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["status"] == "ok"
    assert data[0]["provider_status"] == "ok"

    md_content = (output_dir / "AI_3D_PHASE3A_SF3D_BENCHMARK_REPORT.md").read_text()
    assert "1 successful SF3D run" in md_content


def test_benchmark_runner_unavailable_exits_without_flag(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "test.png").touch()
    output_dir = tmp_path / "outputs"

    with patch("scripts.run_ai3d_benchmark._check_sf3d_enabled", return_value=(False, "SF3D_ENABLED is false")), \
         patch("sys.argv", ["s", "--input-dir", str(input_dir), "--output-dir", str(output_dir), "--modes", "high", "--bg-modes", "off"]):
        with pytest.raises(SystemExit) as exc_info:
            run_benchmark()
    assert exc_info.value.code != 0
    assert not (output_dir / "results.json").exists()


def test_benchmark_runner_allow_unavailable_writes_dryrun(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "test.png").touch()
    output_dir = tmp_path / "outputs"

    unavail_manifest = {
        "session_id": "s", "status": "unavailable", "provider_status": "unavailable",
        "duration_sec": 0, "output_glb_path": None, "output_size_bytes": 0,
        "peak_mem_mb": 0, "worker_metadata": {}, "preprocessing": {},
        "candidate_ranking": [], "resolved_quality": {}, "input_mode": "single_image",
        "candidate_count": 0, "selected_candidate_id": None, "prepared_image_path": None,
        "warnings": [], "errors": [],
    }

    with patch("scripts.run_ai3d_benchmark.generate_ai_3d", return_value=unavail_manifest), \
         patch("scripts.run_ai3d_benchmark.get_mesh_stats", return_value={"vertex_count": 0, "face_count": 0, "mesh_stats_available": False, "geometry_count": 0}), \
         patch("scripts.run_ai3d_benchmark._check_sf3d_enabled", return_value=(False, "SF3D_ENABLED is false")), \
         patch("scripts.run_ai3d_benchmark._git_sha", return_value="abc1234"), \
         patch("sys.argv", ["s", "--input-dir", str(input_dir), "--output-dir", str(output_dir), "--modes", "high", "--bg-modes", "off", "--allow-unavailable"]):
        run_benchmark()

    md_content = (output_dir / "AI_3D_PHASE3A_SF3D_BENCHMARK_REPORT.md").read_text(encoding="utf-8")
    assert "DRY RUN" in md_content
    assert "PROVIDER UNAVAILABLE" in md_content
    assert "Benchmark attempted but no successful SF3D runs" in md_content


def test_benchmark_id_includes_uuid():
    bid = _make_bench_id("stem", "ultra", True)
    parts = bid.split("_")
    # bench_<timestamp>_<uid>_<stem>_<mode>_<bgtag>
    assert len(parts) >= 5
    assert parts[0] == "bench"


def test_external_providers_not_enabled():
    """Confirm benchmark runner never enables Rodin/Meshy/Tripo."""
    import scripts.run_ai3d_benchmark as runner_module
    source = Path(runner_module.__file__).read_text()
    for provider in ("rodin", "meshy", "tripo", "hunyuan"):
        assert provider not in source.lower().replace("# ", "").replace('"rodin"', "").replace('"meshy"', "").replace('"tripo"', "").replace('"hunyuan"', "")


def test_report_encoding_em_dash(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "test.png").touch()
    output_dir = tmp_path / "outputs"

    manifest = _mock_manifest(tmp_path)

    with patch("scripts.run_ai3d_benchmark.generate_ai_3d", return_value=manifest), \
         patch("scripts.run_ai3d_benchmark.get_mesh_stats", return_value={"vertex_count": 0, "face_count": 0, "mesh_stats_available": False, "geometry_count": 0}), \
         patch("scripts.run_ai3d_benchmark._check_sf3d_enabled", return_value=(True, "ok")), \
         patch("scripts.run_ai3d_benchmark._git_sha", return_value="deadbeef"), \
         patch("sys.argv", ["s", "--input-dir", str(input_dir), "--output-dir", str(output_dir), "--modes", "high", "--bg-modes", "off"]):
        run_benchmark()

    content = (output_dir / "AI_3D_PHASE3A_SF3D_BENCHMARK_REPORT.md").read_text(encoding="utf-8")
    assert "\ufffd" not in content   # no replacement character
    assert "\u2014" in content       # proper em dash present
