import pytest
import numpy as np
import cv2
import json
import shutil
from pathlib import Path
from modules.reconstruction_engine.adapter import OpenMVSAdapter
from modules.operations.settings import settings

@pytest.fixture
def temp_workspace(tmp_path):
    # Setup a mock reconstruction workspace
    ws = tmp_path / "recon_ws"
    ws.mkdir()
    images_dir = ws / "images"
    images_dir.mkdir()
    masks_dir = ws / "masks"
    masks_dir.mkdir()
    
    # Create some mock images and masks
    for i in range(25):
        frame_name = f"frame_{i:04d}.jpg"
        img_path = images_dir / frame_name
        # Simple white image
        cv2.imwrite(str(img_path), np.full((100, 100, 3), 255, dtype=np.uint8))
        
        mask_path = masks_dir / f"{frame_name}.png"
        # 80x80 white box in 100x100 mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 10:90] = 255
        cv2.imwrite(str(mask_path), mask)
        
    capture_dir = tmp_path / "capture"
    frames_dir = capture_dir / "frames"
    frames_dir.mkdir(parents=True)
    orig_masks_dir = frames_dir / "masks"
    orig_masks_dir.mkdir()
    
    input_frames = []
    for i in range(25):
        frame_name = f"frame_{i:04d}.jpg"
        src_path = frames_dir / frame_name
        cv2.imwrite(str(src_path), np.full((100, 100, 3), 255, dtype=np.uint8))
        input_frames.append(str(src_path))
        
        # Create original mask
        orig_mask_path = orig_masks_dir / f"{frame_name}.png"
        orig_mask = np.full((100, 100), 255, dtype=np.uint8)
        cv2.imwrite(str(orig_mask_path), orig_mask)
        
        meta_path = orig_masks_dir / f"frame_{i:04d}.json"
        meta = {
            "is_clipped": False,
            "support_suspected": False,
            "occupancy": 0.64
        }
        # Mark some as clipped/contaminated
        if i >= 20:
            meta["is_clipped"] = True
        if i == 19:
            meta["support_suspected"] = True
            
        with open(meta_path, "w") as f:
            json.dump(meta, f)
            
    return {
        "ws": ws,
        "input_frames": input_frames,
        "prep": {"masks_dir": masks_dir},
        "orig_masks_dir": orig_masks_dir
    }

def test_mask_erosion_reduces_area(temp_workspace):
    prep = temp_workspace["prep"]
    input_frames = temp_workspace["input_frames"]
    masks_dir = prep["masks_dir"]
    
    mask_path = masks_dir / "frame_0000.jpg.png"
    initial_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    initial_area = np.sum(initial_mask > 0)
    
    adapter = OpenMVSAdapter()
    import io
    log = io.StringIO()
    
    # Configure erosion
    settings.texture_mask_erode_px = 5
    adapter._refine_texture_masks(prep, input_frames, log)
    
    refined_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    refined_area = np.sum(refined_mask > 0)
    
    assert refined_area < initial_area
    assert refined_area > 0 # Should not be completely gone

def test_mask_frame_rejection(temp_workspace):
    prep = temp_workspace["prep"]
    input_frames = temp_workspace["input_frames"]
    masks_dir = prep["masks_dir"]
    
    adapter = OpenMVSAdapter()
    import io
    log = io.StringIO()
    
    settings.texture_reject_subject_clipped = True
    settings.texture_reject_support_contamination = True
    settings.texture_min_clean_frames = 10 # We have 19 clean frames (0-18)
    
    adapter._refine_texture_masks(prep, input_frames, log)
    
    # Frame 20+ should be blanked (clipped)
    clipped_mask_path = masks_dir / "frame_0020.jpg.png"
    mask = cv2.imread(str(clipped_mask_path), cv2.IMREAD_GRAYSCALE)
    assert np.sum(mask > 0) == 0
    
    # Frame 19 should be blanked (support)
    support_mask_path = masks_dir / "frame_0019.jpg.png"
    mask = cv2.imread(str(support_mask_path), cv2.IMREAD_GRAYSCALE)
    assert np.sum(mask > 0) == 0
    
    # Frame 0 should still be visible (eroded but not blanked)
    clean_mask_path = masks_dir / "frame_0000.jpg.png"
    mask = cv2.imread(str(clean_mask_path), cv2.IMREAD_GRAYSCALE)
    assert np.sum(mask > 0) > 0

def test_mask_fallback_logic(temp_workspace):
    prep = temp_workspace["prep"]
    input_frames = temp_workspace["input_frames"]
    masks_dir = prep["masks_dir"]
    
    adapter = OpenMVSAdapter()
    import io
    log = io.StringIO()
    
    # We have 19 clean frames. Set min_clean to 20 to trigger fallback.
    settings.texture_min_clean_frames = 20
    
    adapter._refine_texture_masks(prep, input_frames, log)
    
    # Frame 20 (clipped) should NOT be blanked because of fallback
    clipped_mask_path = masks_dir / "frame_0020.jpg.png"
    mask = cv2.imread(str(clipped_mask_path), cv2.IMREAD_GRAYSCALE)
    assert np.sum(mask > 0) > 0
    
    assert "fallback: too few clean frames" in log.getvalue()

def test_original_masks_not_overwritten(temp_workspace):
    prep = temp_workspace["prep"]
    input_frames = temp_workspace["input_frames"]
    orig_masks_dir = temp_workspace["orig_masks_dir"]
    
    adapter = OpenMVSAdapter()
    import io
    log = io.StringIO()
    
    settings.texture_mask_erode_px = 5
    adapter._refine_texture_masks(prep, input_frames, log)
    
    # Check original mask for frame 0
    orig_mask_path = orig_masks_dir / "frame_0000.jpg.png"
    orig_mask = cv2.imread(str(orig_mask_path), cv2.IMREAD_GRAYSCALE)
    # Original mask should be full white (area 10000)
    assert np.sum(orig_mask > 0) == 10000
