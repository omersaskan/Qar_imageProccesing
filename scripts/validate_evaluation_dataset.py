
import os
import json
import cv2
from pathlib import Path
from typing import List, Dict, Any

def validate_dataset(root: Path) -> Dict[str, Any]:
    errors = []
    warnings = []
    
    videos_dir = root / "videos"
    masks_dir = root / "ground_truth_masks"
    metadata_dir = root / "metadata"
    
    # 1. Verify directories exist
    for d in [videos_dir, masks_dir, metadata_dir]:
        if not d.exists():
            errors.append(f"Missing directory: {d.name}")
            
    if errors:
        return {"status": "failed", "errors": errors}
        
    # 2. Check videos count
    video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.mov"))
    if len(video_files) < 5:
        warnings.append(f"Less than 5 videos found (found {len(video_files)})")
        
    # 3. Verify metadata for every video
    metadata_files = list(metadata_dir.glob("*.json"))
    for meta_file in metadata_files:
        try:
            with open(meta_file, "r") as f:
                data = json.load(f)
                
            video_id = data.get("video_id")
            if not video_id:
                errors.append(f"Metadata {meta_file.name} missing video_id")
                continue
                
            # Check validation frames count
            frames = data.get("validation_frames", [])
            if len(frames) < 3:
                warnings.append(f"Video {video_id} has less than 3 validation frames")
                
            # Verify gt_mask_path exists and is valid
            for frame in frames:
                gt_path_str = frame.get("gt_mask_path")
                if not gt_path_str:
                    errors.append(f"Video {video_id} frame missing gt_mask_path")
                    continue
                    
                # The path in JSON is usually relative to evaluation root or just filename
                # Let's assume it's either absolute or relative to ground_truth_masks
                gt_path = masks_dir / Path(gt_path_str).name
                if not gt_path.exists():
                    errors.append(f"Video {video_id} mask missing: {gt_path_str}")
                else:
                    # Check if readable binary PNG
                    img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        errors.append(f"Video {video_id} mask unreadable: {gt_path.name}")
                    else:
                        unique_vals = len(os.unique(img.flatten())) if hasattr(os, "unique") else len(set(img.flatten()))
                        if unique_vals > 2:
                            warnings.append(f"Video {video_id} mask {gt_path.name} is not strictly binary")
                            
        except Exception as e:
            errors.append(f"Error parsing metadata {meta_file.name}: {e}")
            
    status = "passed" if not errors else "failed"
    return {
        "status": status,
        "video_count": len(video_files),
        "metadata_count": len(metadata_files),
        "errors": errors,
        "warnings": warnings
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate Evaluation Dataset")
    parser.add_argument("--root", required=True, help="Root directory of evaluation dataset")
    args = parser.parse_args()
    
    result = validate_dataset(Path(args.root))
    print(json.dumps(result, indent=2))
    
    if result["status"] == "failed":
        exit(1)
