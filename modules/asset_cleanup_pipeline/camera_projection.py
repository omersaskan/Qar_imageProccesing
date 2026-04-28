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

def _resolve_dense_sparse_dir(workspace_path: Path) -> Optional[Path]:
    """
    Resolves the undistorted dense/sparse model directory.
    Supports both call styles:
      - workspace_path = attempt_dir  -> attempt_dir/dense/sparse
      - workspace_path = dense_dir    -> dense_dir/sparse (if sparse has flat files)
    Returns None if no undistorted model is found.
    """
    # Case 1: workspace_path is attempt_dir -> look in dense/sparse
    dense_sparse = workspace_path / "dense" / "sparse"
    if dense_sparse.exists():
        cam_bin = dense_sparse / "cameras.bin"
        img_bin = dense_sparse / "images.bin"
        if cam_bin.exists() and img_bin.exists():
            return dense_sparse

    # Case 2: workspace_path is already dense/ -> look in sparse/
    # Dense/sparse has flat files (cameras.bin directly in sparse/), not numbered subdirs
    sparse_dir = workspace_path / "sparse"
    if sparse_dir.exists():
        cam_bin = sparse_dir / "cameras.bin"
        img_bin = sparse_dir / "images.bin"
        # Only treat as undistorted if files exist directly (not in numbered subdirs)
        # Undistorted models have flat files; original models have sparse/0/, sparse/1/ etc.
        subdirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
        if cam_bin.exists() and img_bin.exists() and not subdirs:
            return sparse_dir

    return None


def _load_cameras_from_model_dir(model_dir: Path, model_space: str) -> List[Dict[str, Any]]:
    """
    Loads camera data from a single COLMAP sparse model directory.
    """
    cam_bin = model_dir / "cameras.bin"
    img_bin = model_dir / "images.bin"
    if not cam_bin.exists() or not img_bin.exists():
        return []

    cameras_data = []
    cams = read_colmap_cameras_bin(cam_bin)
    imgs = read_colmap_images_bin(img_bin)

    # Determine camera model type for logging
    cam_model_type = "unknown"
    cam_width = 0
    cam_height = 0
    if cams:
        first_cam = next(iter(cams.values()))
        cam_model_type = first_cam["model"]
        cam_width = first_cam["width"]
        cam_height = first_cam["height"]

    for img_id, img in imgs.items():
        cam = cams.get(img["camera_id"])
        if not cam:
            logger.debug("Image %s (%s) refers to missing camera %s", img_id, img['name'], img['camera_id'])
            continue

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
            "source": "colmap",
            "camera_model_space": model_space,
        })

    if cameras_data:
        logger.info(
            "Loaded %d cameras from COLMAP model: %s "
            "(selected_camera_model_space=%s, camera_model_type=%s, width=%d, height=%d)",
            len(cameras_data), model_dir, model_space, cam_model_type, cam_width, cam_height
        )

    return cameras_data


def load_reconstruction_cameras(workspace_path: Path) -> List[Dict[str, Any]]:
    """
    Loads camera projection data from a reconstruction workspace.

    Priority order for mesh-space isolation:
      1. Undistorted dense model (dense/sparse/ or sparse/ if flat) — PINHOLE,
         matches dense/images, fused.ply, meshed-poisson.ply coordinate space.
      2. Original sparse model (sparse/0, sparse/1, ...) — may be RADIAL/OPENCV,
         only used as fallback when no undistorted model exists.

    Supports both call styles:
      - workspace_path = attempt_dir
      - workspace_path = attempt_dir/dense
    """
    # 1. Prefer undistorted dense model (correct for mesh-space isolation)
    dense_sparse = _resolve_dense_sparse_dir(workspace_path)
    if dense_sparse is not None:
        cameras_data = _load_cameras_from_model_dir(dense_sparse, "undistorted_dense")
        if cameras_data:
            return cameras_data
        logger.warning("Dense sparse model found at %s but produced no cameras", dense_sparse)

    # 2. Fallback: original sparse models (sparse/0, sparse/1, ...)
    # Try workspace_path/sparse first, then workspace_path/dense/sparse (numbered subdirs)
    for base in [workspace_path, workspace_path / "dense"]:
        sparse_dir = base / "sparse"
        if not sparse_dir.exists():
            continue

        def get_model_size(d: Path) -> int:
            img_bin = d / "images.bin"
            return img_bin.stat().st_size if img_bin.exists() else 0

        model_dirs = sorted(
            [d for d in sparse_dir.iterdir() if d.is_dir()],
            key=get_model_size, reverse=True
        )
        if not model_dirs:
            # Files might be directly in sparse_dir
            model_dirs = [sparse_dir]

        for model_dir in model_dirs:
            cameras_data = _load_cameras_from_model_dir(model_dir, "original_sparse")
            if cameras_data:
                logger.warning(
                    "Using ORIGINAL sparse model (distorted) for isolation. "
                    "This may cause incorrect mask projections if mesh is in undistorted space."
                )
                return cameras_data

    logger.warning("No COLMAP cameras found in workspace: %s", workspace_path)
    return []

def load_reconstruction_masks(
    workspace_path: Path,
    camera_names: List[str],
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Loads mask images corresponding to the camera names.

    Supports both call styles:
      - workspace_path = attempt_dir
      - workspace_path = attempt_dir/dense

    For dense-space isolation, dense/stereo/masks is preferred because
    those masks are already resized to match dense image dimensions.

    If expected_width/expected_height are provided (from the loaded camera model),
    masks with mismatched dimensions are resized with INTER_NEAREST.

    Supports both naming conventions:
      - frame_0030.jpg -> frame_0030.jpg.png
      - frame_0030.jpg -> frame_0030.png
    """
    masks = {}

    # Build search paths supporting both workspace = attempt_dir and workspace = dense/
    # Priority: dense stereo masks first (correct for mesh-space isolation)
    search_dirs = [
        workspace_path / "dense" / "stereo" / "masks",  # attempt_dir style
        workspace_path / "stereo" / "masks",             # dense/ style
        workspace_path / "dense" / "masks",              # attempt_dir alt
        workspace_path / "masks",                        # both: feature masks
    ]

    # Determine which search dir we actually loaded from (for logging)
    selected_mask_dir = None
    resize_count = 0

    for name in camera_names:
        found = False
        stem = Path(name).stem

        for d in search_dirs:
            if not d.exists():
                continue

            # Try both naming conventions:
            #   frame_0000.jpg.png  (COLMAP stereo mask convention)
            #   frame_0000.png      (stem-only convention)
            candidates = [d / f"{name}.png", d / f"{stem}.png"]
            for cand in candidates:
                if cand.exists():
                    try:
                        mask = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            # Resize if dimensions don't match camera model
                            if expected_width and expected_height:
                                mh, mw = mask.shape[:2]
                                if mw != expected_width or mh != expected_height:
                                    mask = cv2.resize(
                                        mask,
                                        (expected_width, expected_height),
                                        interpolation=cv2.INTER_NEAREST,
                                    )
                                    resize_count += 1
                            masks[name] = mask
                            found = True
                            if selected_mask_dir is None:
                                selected_mask_dir = d
                            break
                    except Exception:
                        pass
            if found:
                break

    # Determine mask space label for logging
    mask_space = "unknown"
    if selected_mask_dir is not None:
        dir_str = str(selected_mask_dir)
        if "stereo" in dir_str:
            mask_space = "dense_stereo"
        elif "dense" in dir_str:
            mask_space = "dense"
        else:
            mask_space = "feature"

    logger.info(
        "Mask loading summary: requested=%d, loaded=%d, mask_space=%s, "
        "mask_dir=%s, resized=%d",
        len(camera_names), len(masks), mask_space,
        selected_mask_dir, resize_count,
    )
    if len(masks) < len(camera_names) * 0.5:
        logger.warning(
            "Low mask coverage: %d/%d. Isolation might be weak.",
            len(masks), len(camera_names),
        )
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
