
import cv2
import numpy as np
from pathlib import Path

def create_tight_mask(image_path: Path, output_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path}")
        return
        
    # Convert to HSV for better segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # The bottle has a blue cap and a clear body.
    # The background is a light green table.
    # Let's try to remove the background.
    
    # Table color is roughly H:60, S:10-30, V:80-90 (in 0-180/0-255/0-255)
    lower_bg = np.array([30, 0, 150])
    upper_bg = np.array([100, 100, 255])
    
    bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)
    
    # Invert to get foreground
    fg_mask = cv2.bitwise_not(bg_mask)
    
    # Refine with morphological ops
    kernel = np.ones((5,5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Keep only the largest contour (the bottle)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(fg_mask)
        cv2.drawContours(mask, [largest_cnt], -1, 255, -1)
        
        # Save GT mask
        cv2.imwrite(str(output_path), mask)
        print(f"Generated GT mask at {output_path}")
    else:
        print(f"No contours found for {image_path}")

if __name__ == "__main__":
    eval_root = Path("datasets/evaluation")
    gt_dir = eval_root / "ground_truth_masks"
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    frames_to_gt = [0, 10, 20]
    video_id = "cap_29ab6fa1"
    
    source_frames_dir = Path("data/captures/cap_29ab6fa1/frames")
    
    for f_idx in frames_to_gt:
        frame_name = f"frame_{f_idx:04d}.jpg"
        src_path = source_frames_dir / frame_name
        dest_path = gt_dir / f"{video_id}_f{f_idx}.png"
        
        if src_path.exists():
            create_tight_mask(src_path, dest_path)
        else:
            print(f"Source frame {src_path} not found")
