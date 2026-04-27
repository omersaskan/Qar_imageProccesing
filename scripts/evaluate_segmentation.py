
import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any

def calculate_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)

def calculate_leakage_ratio(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Calculate background leakage ratio.
    Leakage is defined as pixels in pred_mask that are NOT in gt_mask,
    relative to the total area of the gt_mask.
    """
    leakage = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    gt_area = gt_mask.sum()
    if gt_area == 0:
        return 1.0 if pred_mask.sum() > 0 else 0.0
    return float(leakage / gt_area)

def evaluate_masks(gt_dir: Path, pred_dir: Path) -> Dict[str, Any]:
    """
    Evaluate predicted masks against ground truth.
    Expects matching filenames in both directories.
    """
    results = []
    gt_files = list(gt_dir.glob("*.png"))
    
    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
            
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 127
        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE) > 127
        
        if gt_mask.shape != pred_mask.shape:
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            
        iou = calculate_iou(gt_mask, pred_mask)
        leakage = calculate_leakage_ratio(gt_mask, pred_mask)
        
        results.append({
            "file": gt_path.name,
            "iou": iou,
            "leakage_ratio": leakage
        })
        
    if not results:
        return {"status": "error", "message": "No matching mask files found"}
        
    avg_iou = np.mean([r["iou"] for r in results])
    avg_leakage = np.mean([r["leakage_ratio"] for r in results])
    
    return {
        "status": "success",
        "sample_count": len(results),
        "avg_iou": float(avg_iou),
        "avg_leakage_ratio": float(avg_leakage),
        "per_file_results": results
    }

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Mask Accuracy")
    parser.add_argument("--gt", required=True, help="Ground truth masks directory")
    parser.add_argument("--pred", required=True, help="Predicted masks directory")
    parser.add_argument("--output", help="Optional path to save metrics JSON")
    args = parser.parse_args()
    
    metrics = evaluate_masks(Path(args.gt), Path(args.pred))
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output}")
    else:
        print(json.dumps(metrics, indent=2))
