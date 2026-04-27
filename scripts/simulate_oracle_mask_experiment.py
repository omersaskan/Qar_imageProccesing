
import os
import json
import shutil
from pathlib import Path

def simulate():
    job_id = "cap_29ab6fa1"
    eval_root = Path("datasets/evaluation")
    gt_masks_dir = eval_root / "ground_truth_masks"
    
    # Frames that exist in the sparse model
    sparse_frames = [0, 2, 4]
    
    # 1. Create a simulated Oracle mask directory
    sim_oracle_dir = Path(f"data/captures/{job_id}/frames/oracle_masks")
    sim_oracle_dir.mkdir(parents=True, exist_ok=True)
    
    # We'll use the frame_0000 GT mask for all 3 as a placeholder for the experiment
    gt_source = gt_masks_dir / f"{job_id}_f0.png"
    
    for f_idx in sparse_frames:
        target_name = f"frame_{f_idx:04d}.png"
        shutil.copy2(gt_source, sim_oracle_dir / target_name)
        
        # Create metadata indicating sam2 was used (to test detection logic)
        meta = {
            "segmentation_method": "sam2",
            "mask_confidence": 0.98,
        }
        with open(sim_oracle_dir / f"frame_{f_idx:04d}.json", "w") as f:
            json.dump(meta, f)
            
    print(f"Simulated Oracle masks created in {sim_oracle_dir} for frames {sparse_frames}")

if __name__ == "__main__":
    simulate()
