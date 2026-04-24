import pytest
import trimesh
import numpy as np
from modules.export_pipeline.glb_exporter import GLBExporter

def test_glb_export_forces_texture_visuals_fallback(tmp_path):
    exporter = GLBExporter()
    
    # Create a dummy mesh with UVs but an incompatible material
    mesh = trimesh.creation.box()
    
    # Assign UVs
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.random.rand(len(mesh.vertices), 2)
    )
    
    # Assign an incompatible material (e.g. SimpleMaterial which lacks PBR slots)
    incompatible_mat = trimesh.visual.material.SimpleMaterial(
        diffuse=(255, 0, 0, 255)
    )
    mesh.visual.material = incompatible_mat
    
    mesh_path = tmp_path / "dummy.ply"
    mesh.export(str(mesh_path))
    
    texture_path = tmp_path / "dummy_tex.png"
    from PIL import Image
    img = Image.new("RGBA", (10, 10), color="red")
    img.save(texture_path)
    
    output_path = tmp_path / "output.glb"
    
    # Export it
    res = exporter.export_to_glb(str(mesh_path), str(texture_path), str(output_path))
    
    # Load back to check material
    out_scene = trimesh.load(str(output_path))
    if isinstance(out_scene, trimesh.Scene):
        out_mesh = list(out_scene.geometry.values())[0]
    else:
        out_mesh = out_scene
        
    assert hasattr(out_mesh.visual, "material")
    mat = out_mesh.visual.material
    # The new material should be a PBRMaterial since the fallback was triggered
    assert isinstance(mat, trimesh.visual.material.PBRMaterial)
    assert hasattr(mat, "baseColorTexture")
