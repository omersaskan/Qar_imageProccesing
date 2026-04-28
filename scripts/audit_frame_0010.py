import cv2
import numpy as np
from pathlib import Path

def create_overlay(frame, mask, color=(0, 255, 0), alpha=0.5):
    if mask is None or mask.size == 0:
        return frame.copy()
    overlay = frame.copy()
    mask_bool = mask > 127
    overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
    return overlay

def main():
    audit_dir = Path("results/frame_0010_audit")
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    frame_path = Path("data/captures/cap_29ab6fa1/frames/frame_0010.jpg")
    gt_path = Path("datasets/evaluation/ground_truth_masks/cap_29ab6fa1_f10.png")
    
    # We'll use the masks from the best runs
    # Tiny best: manual_first_frame_box (from results/sam2_sweep_tiny_cap_29ab6fa1)
    # Large best: manual_first_frame_box (from results/sam2_live_large_cap_29ab6fa1)
    
    # Wait, my sweep script doesn't save mode-specific folders yet.
    # I'll re-run just these two for the audit.
    
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"Error: Frame not found at {frame_path}")
        return
        
    gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    
    cv2.imwrite(str(audit_dir / "frame_0010_raw.jpg"), frame)
    
    if gt_mask is not None:
        gt_overlay = create_overlay(frame, gt_mask, color=(0, 255, 0)) # Green for GT
        cv2.imwrite(str(audit_dir / "frame_0010_gt_overlay.jpg"), gt_overlay)
    
    # Helper to run a single inference and get a mask
    from modules.operations.settings import settings
    from modules.capture_workflow.object_masker import ObjectMasker
    from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper
    from modules.ai_segmentation.prompting import generate_prompts
    from unittest.mock import patch
    
    # 1. Legacy Mask
    with patch.object(settings, "segmentation_method", "legacy"), \
         patch.object(settings, "sam2_enabled", False):
        masker = ObjectMasker()
        mask, _ = masker.generate_mask(frame)
        cv2.imwrite(str(audit_dir / "frame_0010_legacy_overlay.jpg"), create_overlay(frame, mask, color=(255, 0, 0))) # Red
        
    # 2. SAM2 Tiny (manual_first_frame_box)
    # We need to load frame_0000 GT to get the manual box for the FIRST frame 
    # but for frame_0010 it will just use the default mode.
    # In my script, manual_first_frame_box falls back to center_box for i > 0.
    
    def get_sam2_mask(model_cfg, checkpoint, mode):
        with patch.object(settings, "sam2_enabled", True), \
             patch.object(settings, "sam2_model_cfg", model_cfg), \
             patch.object(settings, "sam2_checkpoint", checkpoint), \
             patch.object(settings, "sam2_device", "cuda"):
            sam2 = SAM2Wrapper()
            h, w = frame.shape[:2]
            prompt = generate_prompts((h, w), mode=mode)
            mask = sam2.segment_frame(frame, prompt)
            return mask

    tiny_mask = get_sam2_mask("configs/sam2.1/sam2.1_hiera_t.yaml", "models/sam2/sam2.1_hiera_tiny.pt", "center_box")
    if tiny_mask is not None:
        cv2.imwrite(str(audit_dir / "frame_0010_sam2_tiny_overlay.jpg"), create_overlay(frame, tiny_mask, color=(0, 0, 255))) # Blue
        
    large_mask = get_sam2_mask("configs/sam2.1/sam2.1_hiera_l.yaml", "models/sam2/sam2.1_hiera_large.pt", "center_box")
    if large_mask is not None:
        cv2.imwrite(str(audit_dir / "frame_0010_sam2_large_overlay.jpg"), create_overlay(frame, large_mask, color=(255, 255, 0))) # Yellow

if __name__ == "__main__":
    main()
