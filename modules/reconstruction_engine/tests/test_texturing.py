import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
from modules.reconstruction_engine.openmvs_texturer import OpenMVSTexturer

def test_texturer_skips_when_binaries_missing(tmp_path):
    texturer = OpenMVSTexturer(bin_dir=str(tmp_path))
    
    out_dir = tmp_path / "texturing_out"
    out_dir.mkdir()
    
    with pytest.raises(RuntimeError, match="OpenMVS binaries missing"):
        texturer.run_texturing(
            colmap_workspace=tmp_path / "colmap",
            dense_workspace=tmp_path / "dense",
            selected_mesh=str(tmp_path / "mesh.ply"),
            output_dir=out_dir
        )

@patch("subprocess.Popen")
def test_texturer_executes_pipeline_safely(mock_popen, tmp_path):
    texturer = OpenMVSTexturer(bin_dir=str(tmp_path))
    
    # Mock binary existence
    texturer._interface_colmap.touch()
    texturer._texture_mesh.touch()
    
    # Mock subprocess return code
    mock_proc = mock_popen.return_value
    mock_proc.stdout = []
    mock_proc.wait.return_value = 0
    mock_proc.returncode = 0
    
    out_dir = tmp_path / "texturing_out"
    out_dir.mkdir()
    
    # Mock the outputs that TextureMesh would generate
    (out_dir / "scene.mvs").touch()
    (out_dir / "textured_model.obj").touch()
    (out_dir / "textured_model_material_0_map_Kd.png").touch()
    
    result = texturer.run_texturing(
        colmap_workspace=tmp_path / "colmap",
        dense_workspace=tmp_path / "dense",
        selected_mesh=str(tmp_path / "mesh.ply"),
        output_dir=out_dir
    )
    
    assert "textured_mesh_path" in result
    assert "textured_model.obj" in result["textured_mesh_path"]
    assert len(result["texture_atlas_paths"]) == 1
    assert result["texturing_engine"] == "openmvs"
