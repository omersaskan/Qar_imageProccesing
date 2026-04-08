import trimesh
import numpy as np
import cv2
import xatlas
from pathlib import Path
import os
import io

class TextureBaker:
    """
    Bakes vertex colors from an untextured mesh (e.g. from COLMAP output)
    into a UV-mapped texture atlas, returning an unlit textured mesh.
    """
    def __init__(self, tex_size: int = 2048):
        self.tex_size = tex_size

    def bake_vertex_colors(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Takes a mesh with vertex colors, unwraps it, bakes the colors to a texture,
        and returns a new mesh with a UV map and texture material.
        """
        if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
            return mesh
            
        # Unwrap with xatlas
        vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
        
        # We create an empty texture image
        texture_img = np.zeros((self.tex_size, self.tex_size, 3), dtype=np.uint8)
        
        colors = mesh.visual.vertex_colors[:, :3] # discard alpha
        uv_pixels = (uvs * self.tex_size).astype(np.int32)
        
        # Rebuild faces in UV space
        faces_uvs = uv_pixels[indices]
        new_colors = colors[vmapping]
        faces_colors = new_colors[indices]
        
        # Simple rasterization into texture_img
        for i in range(len(indices)):
            pts = faces_uvs[i]
            # using mean color of the face
            # flip RGB to BGR for cv2
            color_rgb = np.mean(faces_colors[i], axis=0).astype(int).tolist()
            color_bgr = [color_rgb[2], color_rgb[1], color_rgb[0]]
            cv2.fillPoly(texture_img, [pts], color_bgr)
            
        # Convert BGR CV2 image to PIL Image (RGB)
        texture_img_rgb = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(texture_img_rgb)
        
        # Create unlit PBR material with the baked texture
        from trimesh.visual.material import PBRMaterial
        
        # Trimesh expects unlit to be configured by the exporter, 
        # but passing baseColorTexture achieves the primary goal.
        material = PBRMaterial(
            baseColorTexture=pil_img,
            roughnessFactor=1.0,
            metallicFactor=0.0
        )
        
        # Create new visual using the UVs and Material
        new_visual = trimesh.visual.TextureVisuals(
            uv=uvs,
            material=material
        )
        
        # Create new mesh
        new_mesh = trimesh.Trimesh(
            vertices=mesh.vertices[vmapping],
            faces=indices,
            visual=new_visual,
            process=False # avoid merging vertices and messing up UVs
        )
        
        return new_mesh
