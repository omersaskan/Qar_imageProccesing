import trimesh
import numpy as np
from PIL import Image

def test_inspect():
    # Create UV only mesh
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]]
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=[[0,0], [1,0], [0,1]])
    
    # Export to GLB
    glb_data = mesh.export(file_type='glb')
    
    # Reload
    loaded = trimesh.load(trimesh.util.wrap_as_stream(glb_data), file_type='glb', force='scene')
    
    for name, geom in loaded.geometry.items():
        print(f"Geometry: {name}")
        material = getattr(geom.visual, 'material', None)
        if material:
            print(f"  Material type: {type(material)}")
            tex = getattr(material, 'baseColorTexture', None)
            if tex:
                print(f"  baseColorTexture type: {type(tex)}")
                if hasattr(tex, 'size'):
                    print(f"  size: {tex.size}")
                if hasattr(tex, 'shape'):
                    print(f"  shape: {tex.shape}")
            img = getattr(material, 'image', None)
            if img:
                print(f"  image type: {type(img)}")
                if hasattr(img, 'size'):
                    print(f"  size: {img.size}")
                if hasattr(img, 'shape'):
                    print(f"  shape: {img.shape}")

if __name__ == "__main__":
    test_inspect()
