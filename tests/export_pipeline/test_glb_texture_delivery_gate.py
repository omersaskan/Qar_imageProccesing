import os
import pytest
import trimesh
import numpy as np
from pathlib import Path
from PIL import Image
from modules.export_pipeline.glb_exporter import GLBExporter
from modules.operations.settings import settings

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def sample_mesh_path(temp_dir):
    mesh_path = temp_dir / "cube.obj"
    mesh = trimesh.creation.box()
    # Add dummy UVs
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.random.rand(len(mesh.vertices), 2),
        material=trimesh.visual.material.PBRMaterial()
    )
    mesh.export(str(mesh_path))
    return str(mesh_path)

@pytest.fixture
def sample_texture_path(temp_dir):
    tex_path = temp_dir / "texture.png"
    img = Image.new('RGB', (1024, 1024), color=(128, 128, 128))
    img.save(str(tex_path))
    return str(tex_path)

@pytest.fixture
def black_texture_path(temp_dir):
    tex_path = temp_dir / "black.png"
    img = Image.new('RGB', (1024, 1024), color=(0, 0, 0))
    img.save(str(tex_path))
    return str(tex_path)

def test_glb_delivery_gate_success(sample_mesh_path, sample_texture_path, temp_dir):
    exporter = GLBExporter()
    output_path = str(temp_dir / "output_success.glb")
    
    # Enable strict mode
    settings.require_textured_output = True
    
    result = exporter.export(
        mesh_path=sample_mesh_path,
        output_path=output_path,
        texture_path=sample_texture_path,
        profile_name="mobile_high"
    )
    
    assert result["delivery_ready"] is True
    assert result["texture_applied"] is True
    assert result["texture_count"] > 0
    assert result["material_count"] > 0

def test_glb_delivery_gate_fail_no_texture(sample_mesh_path, temp_dir):
    exporter = GLBExporter()
    output_path = str(temp_dir / "output_no_tex.glb")
    
    # Enable strict mode
    settings.require_textured_output = True
    
    result = exporter.export(
        mesh_path=sample_mesh_path,
        output_path=output_path,
        texture_path=None, # Missing texture
        profile_name="mobile_high"
    )
    
    assert result["delivery_ready"] is False
    assert result["export_status"] == "failed_texture_application"

def test_glb_delivery_gate_fail_black_texture(sample_mesh_path, black_texture_path, temp_dir):
    exporter = GLBExporter()
    output_path = str(temp_dir / "output_black.glb")
    
    # Enable strict mode
    settings.require_textured_output = True
    settings.max_empty_texture_ratio = 0.1
    
    result = exporter.export(
        mesh_path=sample_mesh_path,
        output_path=output_path,
        texture_path=black_texture_path,
        profile_name="mobile_high"
    )
    
    assert result["delivery_ready"] is False
    assert result["export_status"] == "failed_texture_quality"
    assert result["highest_black_pixel_ratio"] > 0.9

def test_glb_delivery_gate_fail_missing_uvs(temp_dir, sample_texture_path):
    # Create mesh without UVs
    mesh_path = str(temp_dir / "no_uv.obj")
    mesh = trimesh.creation.box()
    mesh.export(mesh_path)
    
    exporter = GLBExporter()
    output_path = str(temp_dir / "output_no_uv.glb")
    
    settings.require_textured_output = True
    
    result = exporter.export(
        mesh_path=mesh_path,
        output_path=output_path,
        texture_path=sample_texture_path,
        profile_name="mobile_high"
    )
    
    assert result["delivery_ready"] is False
    assert result["texture_applied"] is False
    assert result["export_status"] == "failed_texture_application"
