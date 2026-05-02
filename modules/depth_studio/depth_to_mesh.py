"""Convert depth map to a displaced relief-plane mesh."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np


def build_relief_mesh(
    depth: np.ndarray,
    grid_resolution: int = 256,
    depth_scale: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a subdivided plane displaced by depth.
    Returns (vertices, faces, uvs) as numpy arrays.

    depth: float32 HxW [0,1]
    Returns:
      vertices: (N, 3) float32
      faces:    (M, 3) uint32
      uvs:      (N, 2) float32
    """
    h, w = depth.shape[:2]
    res = grid_resolution

    # Resample depth to grid resolution
    try:
        import cv2
        depth_grid = cv2.resize(depth, (res, res), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        # Fallback: nearest neighbour via numpy
        yi = (np.arange(res) * h / res).astype(int).clip(0, h - 1)
        xi = (np.arange(res) * w / res).astype(int).clip(0, w - 1)
        depth_grid = depth[np.ix_(yi, xi)]

    # Build vertex grid
    xs = np.linspace(-0.5, 0.5, res, dtype=np.float32)
    ys = np.linspace(-0.5, 0.5, res, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    zv = (depth_grid.astype(np.float32) - 0.5) * depth_scale

    vertices = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)

    # UV coords
    u = (xv + 0.5).ravel()
    v = (0.5 - yv).ravel()  # flip Y for image convention
    uvs = np.stack([u, v], axis=1)

    # Build quad faces → two triangles each
    rows, cols = res, res
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            tl = r * cols + c
            tr = tl + 1
            bl = (r + 1) * cols + c
            br = bl + 1
            faces.append([tl, bl, tr])
            faces.append([tr, bl, br])
    faces_arr = np.array(faces, dtype=np.uint32)

    return vertices, faces_arr, uvs


def export_mesh_to_trimesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture_image_path: str,
) -> "trimesh.Trimesh":
    import trimesh
    from PIL import Image

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    img = Image.open(texture_image_path).convert("RGB")
    material = trimesh.visual.texture.SimpleMaterial(image=img)
    tex_vis = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    mesh.visual = tex_vis
    return mesh


def depth_to_glb(
    depth: np.ndarray,
    texture_image_path: str,
    output_glb_path: str,
    grid_resolution: int = 256,
    depth_scale: float = 0.3,
) -> Dict[str, Any]:
    """
    Full pipeline: depth → mesh → textured GLB.
    Returns manifest-ready dict.
    """
    vertices, faces, uvs = build_relief_mesh(depth, grid_resolution, depth_scale)

    try:
        mesh = export_mesh_to_trimesh(vertices, faces, uvs, texture_image_path)
        Path(output_glb_path).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output_glb_path)
        return {
            "status": "ok",
            "glb_path": output_glb_path,
            "mesh_vertex_count": len(vertices),
            "mesh_face_count": len(faces),
            "mesh_mode": "relief_plane",
        }
    except Exception as e:
        return {
            "status": "failed",
            "reason": str(e),
            "glb_path": None,
            "mesh_vertex_count": len(vertices),
            "mesh_face_count": len(faces),
            "mesh_mode": "relief_plane",
        }
