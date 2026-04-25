import os
import cv2
from pathlib import Path

def check_alignment():
    workspace = Path("workspace_quality_test")
    dense_images_dir = workspace / "dense/images"
    raw_masks_dir = workspace / "masks"
    dense_masks_dir = workspace / "dense/stereo/masks"
    
    print(f"--- Alignment Check ---")
    
    # 1. Images
    images = sorted(list(dense_images_dir.glob("*.jpg")))
    print(f"Dense Images: {len(images)}")
    if images:
        img = cv2.imread(str(images[0]))
        print(f"  Sample Image: {images[0].name} | Dim: {img.shape if img is not None else 'Error'}")
    
    # 2. Raw Masks
    raw_masks = sorted(list(raw_masks_dir.glob("*.png")))
    print(f"Raw Masks: {len(raw_masks)}")
    if raw_masks:
        m = cv2.imread(str(raw_masks[0]), 0)
        print(f"  Sample Raw Mask: {raw_masks[0].name} | Dim: {m.shape if m is not None else 'Error'}")
        
    # 3. Dense Masks
    dense_masks = sorted(list(dense_masks_dir.glob("*.png")))
    print(f"Dense Masks: {len(dense_masks)}")
    if dense_masks:
        m = cv2.imread(str(dense_masks[0]), 0)
        print(f"  Sample Dense Mask: {dense_masks[0].name} | Dim: {m.shape if m is not None else 'Error'}")
        
    # 4. Filename Matching Check
    image_names = {f.name for f in images}
    # Raw masks are typically frame_0000.jpg.png
    raw_mask_names = {f.name for f in raw_masks}
    # Dense masks are typically frame_0000.jpg.png
    dense_mask_names = {f.name for f in dense_masks}
    
    match_raw = sum(1 for img_name in image_names if f"{img_name}.png" in raw_mask_names)
    match_dense = sum(1 for img_name in image_names if f"{img_name}.png" in dense_mask_names)
    
    print(f"--- Results ---")
    print(f"Matches with Raw: {match_raw} / {len(images)}")
    print(f"Matches with Dense: {match_dense} / {len(images)}")
    
    if images and dense_masks:
        img = cv2.imread(str(images[0]))
        m = cv2.imread(str(dense_masks[0]))
        if img is not None and m is not None:
            if img.shape[:2] == m.shape[:2]:
                print(f"SUCCESS: Dense mask dimensions MATCH dense image dimensions.")
            else:
                print(f"WARNING: Dimension MISMATCH! Image: {img.shape[:2]}, Mask: {m.shape[:2]}")

if __name__ == "__main__":
    check_alignment()
