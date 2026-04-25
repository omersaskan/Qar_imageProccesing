import pytest
from pathlib import Path
from modules.reconstruction_engine.adapter import OpenMVSCommandBuilder

def test_interface_colmap_command():
    builder = OpenMVSCommandBuilder(r"C:\openmvs")
    workspace = Path("data/reconstructions/test_session/dense")
    output_mvs = workspace / "project.mvs"
    
    cmd = builder.interface_colmap(workspace, output_mvs)
    
    assert "InterfaceCOLMAP" in cmd[0]
    # Required: use -w or --working-folder
    assert "--working-folder" in cmd or "-w" in cmd
    # Required: do not use --working-dir
    assert "--working-dir" not in cmd
    
    # Check that it points to the workspace
    if "--working-folder" in cmd:
        idx = cmd.index("--working-folder")
        assert cmd[idx+1] == str(workspace)

def test_densify_point_cloud_command():
    builder = OpenMVSCommandBuilder(r"C:\openmvs")
    input_mvs = Path("data/reconstructions/test_session/dense/project.mvs")
    output_mvs = Path("data/reconstructions/test_session/dense/project_dense.mvs")
    
    cmd = builder.densify_point_cloud(input_mvs, output_mvs)
    
    assert "DensifyPointCloud" in cmd[0]
    assert "--working-folder" in cmd
    idx = cmd.index("--working-folder")
    assert cmd[idx+1] == str(input_mvs.parent)
    
    assert "-i" in cmd
    idx_i = cmd.index("-i")
    assert cmd[idx_i+1] == str(input_mvs)
    
    assert "-o" in cmd
    idx_o = cmd.index("-o")
    assert cmd[idx_o+1] == str(output_mvs)

def test_reconstruct_mesh_command():
    builder = OpenMVSCommandBuilder(r"C:\openmvs")
    input_mvs = Path("data/reconstructions/test_session/dense/project_dense.mvs")
    output_mesh = Path("data/reconstructions/test_session/dense/project_mesh.ply")
    
    cmd = builder.reconstruct_mesh(input_mvs, output_mesh)
    
    assert "ReconstructMesh" in cmd[0]
    assert "-o" in cmd
    idx = cmd.index("-o")
    # Required: output project_mesh.ply
    assert cmd[idx+1] == str(output_mesh)
    assert cmd[idx+1].endswith("project_mesh.ply")

def test_texture_mesh_command():
    builder = OpenMVSCommandBuilder(r"C:\openmvs")
    input_scene = Path("data/reconstructions/test_session/dense/project_dense.mvs")
    input_mesh = Path("data/reconstructions/test_session/dense/project_mesh.ply")
    output_obj = Path("data/reconstructions/test_session/dense/project_textured.obj")
    
    cmd = builder.texture_mesh(input_scene, input_mesh, output_obj)
    
    assert "TextureMesh" in cmd[0]
    # Required: use --mesh-file project_mesh.ply
    assert "--mesh-file" in cmd
    idx_m = cmd.index("--mesh-file")
    assert cmd[idx_m+1] == str(input_mesh)
    
    # Required: output project_textured.obj
    assert "-o" in cmd
    idx_o = cmd.index("-o")
    assert cmd[idx_o+1] == str(output_obj)
    assert cmd[idx_o+1].endswith(".obj")
    
    # Required: export type obj
    assert "--export-type" in cmd
    idx_e = cmd.index("--export-type")
    assert cmd[idx_e+1] == "obj"
