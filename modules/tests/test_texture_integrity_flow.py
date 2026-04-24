import pytest
import os
import trimesh
import numpy as np
from pathlib import Path
from PIL import Image
from modules.export_pipeline.glb_exporter import GLBExporter
from modules.asset_cleanup_pipeline.remesher import Remesher
from modules.asset_cleanup_pipeline.profiles import CleanupProfile

def test_texture_preservation_export():
    exporter = GLBExporter()
    
    # 1. Create a textured mesh
    mesh = trimesh.creation.icosphere(radius=1.0)
    # Give it some UVs
    uv = np.random.rand(len(mesh.vertices), 2)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
    
    temp_mesh = "test_textured.obj"
    temp_tex = "test_tex.png"
    temp_glb = "test_output.glb"
    
    try:
        mesh.export(temp_mesh)
        Image.new('RGB', (100, 100), color='red').save(temp_tex)
        
        # 2. Export to GLB
        result = exporter.export(
            mesh_path=temp_mesh,
            texture_path=temp_tex,
            output_path=temp_glb
        )
        
        # 3. Verify
        assert result["texture_applied_successfully"] == True
        assert result["has_uv"] == True
        assert result["has_material"] == True
        assert os.path.getsize(temp_glb) > 0
        
    finally:
        for f in [temp_mesh, temp_tex, temp_glb]:
            if os.path.exists(f): os.remove(f)

def test_remesher_uv_preservation():
    remesher = Remesher()
    
    # 1. Create high poly mesh with UVs
    mesh = trimesh.creation.icosphere(subdivisions=4) # ~1280 faces
    uv = np.random.rand(len(mesh.vertices), 2)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
    
    temp_in = "high_poly_uv.obj"
    temp_out = "low_poly_uv.obj"
    mesh.export(temp_in)
    
    from modules.asset_cleanup_pipeline.profiles import CleanupProfileType
    profile = CleanupProfile(
        name=CleanupProfileType.MOBILE_LOW,
        target_polycount=500,
        decimation_ratio=0.5,
        texture_size=1024
    )
    
    try:
        remesher.process(temp_in, temp_out, profile)
        
        # Load and check
        low_mesh = trimesh.load(temp_out)
        has_uv = hasattr(low_mesh.visual, 'uv') and low_mesh.visual.uv is not None
        
        assert has_uv == True
        assert len(low_mesh.faces) <= 600 # Approx match
        
    finally:
        for f in [temp_in, temp_out]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    pytest.main([__file__])
