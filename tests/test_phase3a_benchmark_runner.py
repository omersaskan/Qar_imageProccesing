import pytest
import os
import json
import csv
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from scripts.run_ai3d_benchmark import get_mesh_stats, run_benchmark

def test_get_mesh_stats_missing_file():
    stats = get_mesh_stats("non_existent.glb")
    assert stats["mesh_stats_available"] is False
    assert stats["vertex_count"] == 0

@patch("trimesh.load")
def test_get_mesh_stats_mocked(mock_load):
    import trimesh
    # Mock a trimesh mesh
    mock_mesh = MagicMock()
    mock_mesh.vertices = [1, 2, 3] # length 3
    mock_mesh.faces = [1, 2]       # length 2
    
    # Mock scene behavior
    mock_scene = MagicMock(spec=trimesh.Scene)
    mock_scene.geometry = {"mesh1": mock_mesh}
    
    mock_load.return_value = mock_scene
    
    # Create a dummy file to pass exists check
    dummy_file = "dummy_test.glb"
    Path(dummy_file).touch()
    
    try:
        stats = get_mesh_stats(dummy_file)
        assert stats["mesh_stats_available"] is True
        assert stats["geometry_count"] == 1
        assert stats["vertex_count"] == 3
        assert stats["face_count"] == 2
    finally:
        if os.path.exists(dummy_file):
            os.remove(dummy_file)

def test_benchmark_runner_mocked_execution(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "test.png").touch()
    
    output_dir = tmp_path / "outputs"
    
    # Mock generate_ai_3d
    mock_manifest = {
        "session_id": "test_sess",
        "status": "review",
        "provider_status": "ok",
        "duration_sec": 10.5,
        "output_glb_path": "fake.glb",
        "output_size_bytes": 1000,
        "peak_mem_mb": 500,
        "preprocessing": {"background_removed": True},
        "candidate_ranking": [{"score": 90.0}]
    }
    
    with patch("scripts.run_ai3d_benchmark.generate_ai_3d", return_value=mock_manifest), \
         patch("scripts.run_ai3d_benchmark.get_mesh_stats", return_value={"vertex_count": 0, "face_count": 0, "mesh_stats_available": False, "geometry_count": 0}), \
         patch("sys.argv", ["script.py", "--input-dir", str(input_dir), "--output-dir", str(output_dir), "--modes", "high", "--bg-modes", "on"]):
        
        run_benchmark()
        
    # Verify outputs
    assert (output_dir / "results.json").exists()
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "AI_3D_PHASE3A_SF3D_BENCHMARK_REPORT.md").exists()
    
    with open(output_dir / "results.json") as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]["quality_mode"] == "high"
        assert data[0]["status"] == "review"

def test_benchmark_runner_field_schema(tmp_path):
    # Just verify the row dictionary construction logic indirectly
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "test.png").touch()
    output_dir = tmp_path / "outputs"
    
    mock_manifest = {
        "session_id": "test_sess",
        "status": "review",
        "provider_status": "ok",
        "preprocessing": {"background_removed": True, "mask_source": "rembg", "foreground_ratio_estimate": 0.5},
        "candidate_ranking": [{"score": 90.0}],
        "resolved_quality": {"input_size": 512}
    }
    
    with patch("scripts.run_ai3d_benchmark.generate_ai_3d", return_value=mock_manifest), \
         patch("scripts.run_ai3d_benchmark.get_mesh_stats", return_value={"vertex_count": 100, "face_count": 200, "mesh_stats_available": True, "geometry_count": 1}), \
         patch("sys.argv", ["script.py", "--input-dir", str(input_dir), "--output-dir", str(output_dir), "--modes", "high", "--bg-modes", "on"]):
        
        run_benchmark()
        
    with open(output_dir / "results.csv", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        assert "benchmark_id" in row
        assert "score" in row
        assert "vertex_count" in row
        assert row["vertex_count"] == "100"
        assert row["bg_removed"] == "True"
