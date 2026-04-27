import os
import struct
import numpy as np
import trimesh
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("camera_projection")

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_colmap_cameras_bin(path: Path) -> Dict[int, Any]:
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = "UNKNOWN"
            if model_id == 0: model_name = "SIMPLE_PINHOLE"; num_params = 3
            elif model_id == 1: model_name = "PINHOLE"; num_params = 4
            elif model_id == 2: model_name = "SIMPLE_RADIAL"; num_params = 4
            elif model_id == 3: model_name = "RADIAL"; num_params = 5
            elif model_id == 4: model_name = "OPENCV"; num_params = 8
            else: num_params = 0 # Defaulting for safety
            
            width = camera_properties[2]
            height = camera_properties[3]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            
            # Simple K matrix construction for PINHOLE/RADIAL (common for mobile)
            # params: [f, cx, cy] or [fx, fy, cx, cy] or [f, cx, cy, k] etc.
            K = np.eye(3)
            if model_name in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
                f, cx, cy = params[0], params[1], params[2]
                K[0, 0] = K[1, 1] = f
                K[0, 2], K[1, 2] = cx, cy
            elif model_name in ["PINHOLE", "OPENCV"]:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                K[0, 0], K[1, 1] = fx, fy
                K[0, 2], K[1, 2] = cx, cy
            else:
                # Radial fallback
                f, cx, cy = params[0], params[1], params[2]
                K[0, 0] = K[1, 1] = f
                K[0, 2], K[1, 2] = cx, cy

            cameras[camera_id] = {
                "id": camera_id,
                "model": model_name,
                "width": width,
                "height": height,
                "params": params,
                "K": K
            }
    return cameras

def read_colmap_images_bin(path: Path) -> Dict[int, Any]:
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.read(24 * num_points2D) # Skip points
            
            R = qvec2rotmat(qvec)
            # Extrinsic matrix [R | t]
            Ext = np.eye(4)
            Ext[:3, :3] = R
            Ext[:3, 3] = tvec
            
            images[image_id] = {
                "id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "R": R,
                "Ext": Ext,
                "camera_id": camera_id,
                "name": image_name
            }
    return images

def load_reconstruction_cameras(workspace_path: Path) -> List[Dict[str, Any]]:
    """
    Loads camera projection data from a reconstruction workspace.
    Supports COLMAP (sparse model) and potentially OpenMVS.
    """
    cameras_data = []
    
    # 1. Try COLMAP sparse models
    sparse_dir = workspace_path / "sparse"
    if sparse_dir.exists():
        # Find best model (usually '0')
        model_dirs = sorted([d for d in sparse_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        if not model_dirs:
            # Check if sparse files are directly in sparse_dir
            model_dirs = [sparse_dir]
            
        for model_dir in model_dirs:
            cam_bin = model_dir / "cameras.bin"
            img_bin = model_dir / "images.bin"
            
            if cam_bin.exists() and img_bin.exists():
                try:
                    cams = read_colmap_cameras_bin(cam_bin)
                    imgs = read_colmap_images_bin(img_bin)
                    
                    for img_id, img in imgs.items():
                        cam = cams.get(img["camera_id"])
                        if not cam: continue
                        
                        # Projection Matrix P = K * [R | t]
                        K = cam["K"]
                        Ext = img["Ext"][:3, :]
                        P = K @ Ext
                        
                        cameras_data.append({
                            "name": img["name"],
                            "P": P,
                            "K": K,
                            "R": img["R"],
                            "t": img["tvec"],
                            "width": cam["width"],
                            "height": cam["height"],
                            "source": "colmap"
                        })
                    
                    if cameras_data:
                        logger.info(f"Loaded {len(cameras_data)} cameras from COLMAP model {model_dir.name}")
                        return cameras_data
                except Exception as e:
                    logger.warning(f"Failed to read COLMAP bin from {model_dir}: {e}")
            
            # TODO: Add .txt support if needed, but .bin is more common in our pipeline
            
    # 2. Try OpenMVS scene data (optional, fallback)
    # OpenMVS usually stores camera data in project.mvs which is a custom binary format
    # For now, we prioritize COLMAP as it's the primary engine.
    
    return cameras_data

def load_reconstruction_masks(workspace_path: Path, camera_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Loads mask images corresponding to the camera names.
    Expected paths:
    - {workspace_path}/masks/{name}.png
    - {workspace_path}/dense/stereo/masks/{name}.png
    """
    masks = {}
    search_dirs = [
        workspace_path / "masks",
        workspace_path / "dense" / "stereo" / "masks",
        workspace_path / "dense" / "masks"
    ]
    
    for name in camera_names:
        found = False
        # Normalize name (remove extension if needed, handle case)
        stem = Path(name).stem
        
        for d in search_dirs:
            if not d.exists(): continue
            
            # Try exact name.png, then stem.png
            candidates = [d / f"{name}.png", d / f"{stem}.png"]
            for cand in candidates:
                if cand.exists():
                    try:
                        mask = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            masks[name] = mask
                            found = True
                            break
                    except Exception:
                        pass
            if found: break
            
    logger.info(f"Loaded {len(masks)} masks for {len(camera_names)} cameras")
    return masks

def project_points(points: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points into 2D image space using projection matrix P.
    Returns:
    - p2d: (N, 2) array of pixel coordinates
    - depth: (N,) array of depths (z in camera coords)
    """
    # Homogeneous 3D points
    p3d_h = np.hstack([points, np.ones((len(points), 1))])
    
    # Project
    p2d_h = p3d_h @ P.T
    
    depth = p2d_h[:, 2]
    
    # Avoid division by zero
    mask = np.abs(depth) > 1e-8
    p2d = np.zeros((len(points), 2))
    p2d[mask, 0] = p2d_h[mask, 0] / depth[mask]
    p2d[mask, 1] = p2d_h[mask, 1] / depth[mask]
    
    return p2d, depth

def compute_component_mask_support(
    component: trimesh.Trimesh, 
    cameras: List[Dict[str, Any]], 
    masks: Dict[str, np.ndarray],
    sample_size: int = 500
) -> Dict[str, Any]:
    """
    Computes semantic mask support for a mesh component.
    """
    if not cameras or not masks:
        return {"avg_support": 0.0, "reason": "no_data"}
        
    # Sample points from the component (vertices + face centers)
    if len(component.vertices) > sample_size // 2:
        v_indices = np.random.choice(len(component.vertices), sample_size // 2, replace=False)
        sample_pts = component.vertices[v_indices]
    else:
        sample_pts = component.vertices
        
    if len(component.faces) > sample_size // 2:
        f_indices = np.random.choice(len(component.faces), sample_size // 2, replace=False)
        centers = component.triangles_center[f_indices]
        sample_pts = np.vstack([sample_pts, centers])
    else:
        sample_pts = np.vstack([sample_pts, component.triangles_center])

    view_supports = []
    total_visible_samples = 0
    total_hits = 0
    supported_view_count = 0
    
    for cam in cameras:
        name = cam["name"]
        if name not in masks: continue
        
        mask = masks[name]
        h, w = mask.shape[:2]
        P = cam["P"]
        
        p2d, depths = project_points(sample_pts, P)
        
        # Filter points in front of camera and within image bounds
        visible_mask = (depths > 0.01) & (p2d[:, 0] >= 0) & (p2d[:, 0] < w) & (p2d[:, 1] >= 0) & (p2d[:, 1] < h)
        visible_count = np.sum(visible_mask)
        
        if visible_count < 10: # Too few samples visible in this view
            continue
            
        # Check hits in mask
        hits = 0
        visible_indices = np.where(visible_mask)[0]
        coords = p2d[visible_indices].astype(int)
        
        # Vectorized mask lookup
        pixel_vals = mask[coords[:, 1], coords[:, 0]]
        hits = np.sum(pixel_vals > 127)
        
        ratio = hits / visible_count
        view_supports.append(ratio)
        total_visible_samples += visible_count
        total_hits += hits
        
        if ratio > 0.5:
            supported_view_count += 1

    if not view_supports:
        return {
            "avg_support": 0.0,
            "median_support": 0.0,
            "supported_view_count": 0,
            "hit_ratio": 0.0,
            "reason": "not_visible"
        }
        
    avg_support = float(np.mean(view_supports))
    median_support = float(np.median(view_supports))
    overall_hit_ratio = float(total_hits / max(total_visible_samples, 1))
    
    return {
        "avg_support": avg_support,
        "median_support": median_support,
        "supported_view_count": supported_view_count,
        "hit_ratio": overall_hit_ratio,
        "view_count": len(view_supports),
        "total_samples": len(sample_pts)
    }
