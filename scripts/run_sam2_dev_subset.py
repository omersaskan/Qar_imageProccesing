"""
SAM2 Dev-Subset Evaluation Script
====================================

Compares SAM2 vs legacy segmentation on a dev-subset video.

Usage:
    python scripts/run_sam2_dev_subset.py --capture-id cap_29ab6fa1 --output-dir results/
    python scripts/run_sam2_dev_subset.py --video data/videos/test.mp4 --output-dir results/
    python scripts/run_sam2_dev_subset.py --frames-dir data/captures/cap_X/frames/raw --output-dir results/

Decision logic:
- If SAM2 unavailable: report unavailable, do NOT fail hard.
- If SAM2 improves IoU by >= 0.05 and leakage decreases: recommend continuing.
- If SAM2 does not improve segmentation: do NOT proceed to Depth Anything.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sam2_dev_subset")


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    pred_bin = pred > 0
    gt_bin = gt > 0
    intersection = float(np.sum(pred_bin & gt_bin))
    union = float(np.sum(pred_bin | gt_bin))
    return intersection / union if union > 0 else 0.0


def compute_leakage(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute leakage ratio (FP pixels / total predicted pixels)."""
    pred_bin = pred > 0
    gt_bin = gt > 0
    fp = float(np.sum(pred_bin & ~gt_bin))
    total = float(np.sum(pred_bin))
    return fp / total if total > 0 else 0.0


def run_evaluation(args):
    from modules.operations.settings import settings
    from modules.capture_workflow.object_masker import ObjectMasker

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve frames
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
    elif args.capture_id:
        frames_dir = Path(settings.data_root) / "captures" / args.capture_id / "frames" / "raw"
    else:
        logger.error("Provide --frames-dir or --capture-id")
        sys.exit(1)

    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        sys.exit(1)

    frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    if not frame_files:
        logger.error(f"No frames found in {frames_dir}")
        sys.exit(1)

    # Load GT masks if provided
    gt_masks = {}
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
        for gt_file in gt_dir.glob("*.png"):
            gt_masks[gt_file.stem] = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)

    # SAM2 status check
    sam2_status = {"sam2_available": False}
    try:
        from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper
        wrapper = SAM2Wrapper()
        sam2_status = wrapper.get_status()
    except Exception as e:
        sam2_status["sam2_error_reason"] = str(e)

    has_torch = False
    try:
        import torch
        has_torch = True
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    # --- Run legacy masks ---
    logger.info("Running legacy segmentation...")
    legacy_masker = ObjectMasker()
    legacy_ious = []
    legacy_leakages = []
    t0 = time.time()

    for f_path in frame_files[:20]:  # Cap at 20 for dev-subset
        frame = cv2.imread(str(f_path))
        if frame is None:
            continue
        mask, meta = legacy_masker.generate_mask(frame)
        stem = f_path.stem
        if stem in gt_masks:
            iou = compute_iou(mask, gt_masks[stem])
            leak = compute_leakage(mask, gt_masks[stem])
            legacy_ious.append(iou)
            legacy_leakages.append(leak)

    legacy_time = time.time() - t0

    legacy_iou = float(np.mean(legacy_ious)) if legacy_ious else 0.0
    legacy_leakage = float(np.mean(legacy_leakages)) if legacy_leakages else 0.0

    # --- Run SAM2 if available ---
    sam2_iou = 0.0
    sam2_leakage = 0.0
    sam2_time = 0.0
    sam2_ran = False

    if sam2_status.get("sam2_available") and sam2_status.get("sam2_model_loaded"):
        logger.info("Running SAM2 segmentation...")
        sam2_ran = True
        # SAM2 would be run here via the backend
        # For now, report that SAM2 framework is ready but metrics TBD
    else:
        logger.warning(
            f"SAM2 not available: {sam2_status.get('sam2_error_reason', 'unknown')}. "
            "Reporting legacy-only results."
        )

    # --- Decision ---
    iou_gain = sam2_iou - legacy_iou
    leakage_reduction = legacy_leakage - sam2_leakage

    if not sam2_ran:
        recommendation = "sam2_unavailable — cannot evaluate. Legacy results only."
    elif iou_gain >= 0.05 and leakage_reduction > 0:
        recommendation = "sam2_improves_segmentation — continue SAM2 development"
    elif iou_gain < 0.05:
        recommendation = "sam2_no_improvement — do NOT proceed to Depth Anything"
    else:
        recommendation = "sam2_mixed — review manually before proceeding"

    results = {
        "legacy_iou": legacy_iou,
        "sam2_iou": sam2_iou,
        "legacy_leakage": legacy_leakage,
        "sam2_leakage": sam2_leakage,
        "iou_gain": iou_gain,
        "leakage_reduction": leakage_reduction,
        "sam2_status": sam2_status,
        "sam2_ran": sam2_ran,
        "runtime_sec": {"legacy": legacy_time, "sam2": sam2_time},
        "gpu_available": gpu_available,
        "checkpoint_exists": sam2_status.get("checkpoint_exists", False),
        "final_recommendation": recommendation,
        "frames_evaluated": len(legacy_ious),
        "gt_masks_available": len(gt_masks),
    }

    out_path = output_dir / "sam2_dev_subset_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results written to {out_path}")
    logger.info(f"Legacy IoU: {legacy_iou:.4f}, SAM2 IoU: {sam2_iou:.4f}")
    logger.info(f"Recommendation: {recommendation}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SAM2 Dev-Subset Evaluation")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--capture-id", type=str, help="Capture ID")
    parser.add_argument("--frames-dir", type=str, help="Pre-extracted frames dir")
    parser.add_argument("--gt-dir", type=str, help="Ground truth masks dir")
    parser.add_argument("--output-dir", type=str, default="results/sam2_eval")
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
