"""
SAM2 Dev-Subset Evaluation Script
====================================

Compares SAM2 (image mode) vs legacy segmentation on a dev-subset.

Usage:
    python scripts/run_sam2_dev_subset.py --frames-dir data/captures/cap_29ab6fa1/frames \
        --gt-dir datasets/evaluation/ground_truth_masks \
        --output-dir results/sam2_live_cap_29ab6fa1

Decision logic:
- If SAM2 unavailable: report unavailable, do NOT fail hard.
- If SAM2 improves IoU by >= 0.05 and leakage decreases: recommend continuing.
- If SAM2 does not improve: do NOT proceed to Depth Anything.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("sam2_dev_subset")


# -------------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------------

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = pred > 0
    gt_bin = gt > 0
    inter = float(np.sum(pred_bin & gt_bin))
    union = float(np.sum(pred_bin | gt_bin))
    return inter / union if union > 0 else 0.0


def compute_leakage(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = pred > 0
    gt_bin = gt > 0
    fp = float(np.sum(pred_bin & ~gt_bin))
    total = float(np.sum(pred_bin))
    return fp / total if total > 0 else 0.0


def mask_area(mask: np.ndarray) -> int:
    return int(np.sum(mask > 0))


def mask_centroid(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def mask_bbox_center(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float((xs.min() + xs.max()) / 2.0), float((ys.min() + ys.max()) / 2.0)


def compute_temporal_stability(masks_list):
    """Compute temporal stability metrics across a list of masks."""
    areas = [mask_area(m) for m in masks_list]
    centroids = [mask_centroid(m) for m in masks_list]
    bbox_centers = [mask_bbox_center(m) for m in masks_list]
    empty_count = sum(1 for a in areas if a == 0)

    # Area std
    area_std = float(np.std(areas)) if len(areas) > 1 else 0.0

    # Centroid jitter (mean frame-to-frame distance)
    centroid_dists = []
    for i in range(1, len(centroids)):
        if centroids[i] is not None and centroids[i - 1] is not None:
            dx = centroids[i][0] - centroids[i - 1][0]
            dy = centroids[i][1] - centroids[i - 1][1]
            centroid_dists.append(float(np.sqrt(dx ** 2 + dy ** 2)))
    centroid_jitter = float(np.mean(centroid_dists)) if centroid_dists else 0.0

    # Bbox center jitter
    bbox_dists = []
    for i in range(1, len(bbox_centers)):
        if bbox_centers[i] is not None and bbox_centers[i - 1] is not None:
            dx = bbox_centers[i][0] - bbox_centers[i - 1][0]
            dy = bbox_centers[i][1] - bbox_centers[i - 1][1]
            bbox_dists.append(float(np.sqrt(dx ** 2 + dy ** 2)))
    bbox_center_jitter = float(np.mean(bbox_dists)) if bbox_dists else 0.0

    return {
        "mask_area_std": area_std,
        "centroid_jitter": centroid_jitter,
        "bbox_center_jitter": bbox_center_jitter,
        "empty_mask_count": empty_count,
        "frame_count": len(masks_list),
    }


# -------------------------------------------------------------------
# GT name mapping
# -------------------------------------------------------------------

def build_gt_map(gt_dir: Path, capture_prefix: str = "cap_29ab6fa1"):
    """
    Build mapping from frame stem (e.g. 'frame_0000') to GT mask.

    GT files are named like:  cap_29ab6fa1_f0.png  (frame index 0)
    Frame files are named:    frame_0000.jpg       (frame index 0)
    """
    gt_masks = {}
    for gt_file in sorted(gt_dir.glob("*.png")):
        # Extract frame index from GT name: cap_29ab6fa1_f10.png → 10
        m = re.search(r"_f(\d+)\.png$", gt_file.name)
        if m:
            idx = int(m.group(1))
            frame_stem = f"frame_{idx:04d}"
            gt_masks[frame_stem] = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
            logger.info(f"GT mapped: {gt_file.name} → {frame_stem}")
    return gt_masks


# -------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------

def run_evaluation(args):
    from modules.operations.settings import settings
    from modules.capture_workflow.object_masker import ObjectMasker

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_masks_dir = output_dir / "legacy_masks"
    sam2_masks_dir = output_dir / "sam2_masks"

    # Resolve frames directory
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
    elif args.capture_id:
        frames_dir = (
            Path(settings.data_root)
            / "captures"
            / args.capture_id
            / "frames"
        )
    else:
        logger.error("Provide --frames-dir or --capture-id")
        sys.exit(1)

    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        sys.exit(1)

    frame_files = sorted(
        list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
    )
    if not frame_files:
        logger.error(f"No frames found in {frames_dir}")
        sys.exit(1)

    max_frames = min(len(frame_files), 20)
    frame_files = frame_files[:max_frames]
    logger.info(f"Evaluating {len(frame_files)} frames from {frames_dir}")

    # Load GT masks
    gt_masks = {}
    if args.gt_dir:
        gt_masks = build_gt_map(Path(args.gt_dir))
    logger.info(f"GT masks available: {len(gt_masks)} ({list(gt_masks.keys())})")

    # --- SAM2 status ---
    sam2_wrapper = None
    sam2_status = {
        "sam2_available": False,
        "sam2_model_loaded": False,
        "sam2_error_reason": "not checked yet",
    }
    try:
        from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper
        sam2_wrapper = SAM2Wrapper()
        sam2_status = sam2_wrapper.get_status()
    except Exception as e:
        sam2_status["sam2_error_reason"] = str(e)

    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    # --- Run legacy ---
    logger.info("Running legacy segmentation...")
    legacy_masker = ObjectMasker()
    legacy_masks_dir.mkdir(parents=True, exist_ok=True)
    legacy_ious, legacy_leakages = [], []
    legacy_per_frame = []
    legacy_mask_list = []
    t0 = time.time()

    for f_path in frame_files:
        frame = cv2.imread(str(f_path))
        if frame is None:
            continue
        mask, meta = legacy_masker.generate_mask(frame)
        out_path = legacy_masks_dir / f"{f_path.name}.png"
        cv2.imwrite(str(out_path), mask)
        legacy_mask_list.append(mask)

        entry = {
            "frame": f_path.name,
            "mask_path": str(out_path),
            "confidence": meta.get("confidence", 0),
            "occupancy": meta.get("occupancy", 0),
        }
        stem = f_path.stem
        if stem in gt_masks:
            iou = compute_iou(mask, gt_masks[stem])
            leak = compute_leakage(mask, gt_masks[stem])
            legacy_ious.append(iou)
            legacy_leakages.append(leak)
            entry["iou"] = round(iou, 4)
            entry["leakage"] = round(leak, 4)
        legacy_per_frame.append(entry)

    legacy_time = time.time() - t0
    legacy_iou = float(np.mean(legacy_ious)) if legacy_ious else 0.0
    legacy_leakage = float(np.mean(legacy_leakages)) if legacy_leakages else 0.0
    legacy_stability = compute_temporal_stability(legacy_mask_list)
    logger.info(
        f"Legacy: {len(frame_files)} frames in {legacy_time:.2f}s, "
        f"IoU={legacy_iou:.4f}, leakage={legacy_leakage:.4f}"
    )

    # --- Run SAM2 if available ---
    sam2_iou = 0.0
    sam2_leakage = 0.0
    sam2_time = 0.0
    sam2_ran = False
    sam2_ious, sam2_leakages = [], []
    sam2_per_frame = []
    sam2_stability = {
        "mask_area_std": 0, "centroid_jitter": 0,
        "bbox_center_jitter": 0, "empty_mask_count": 0, "frame_count": 0,
    }

    if sam2_wrapper and sam2_wrapper.is_available():
        logger.info("Running SAM2 image-mode segmentation...")
        sam2_masks_dir.mkdir(parents=True, exist_ok=True)
        sam2_ran = True
        sam2_mask_list = []
        t0 = time.time()

        from modules.ai_segmentation.prompting import generate_prompts

        for f_path in frame_files:
            frame = cv2.imread(str(f_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            prompt = generate_prompts(
                frame_shape=(h, w), mode=settings.sam2_prompt_mode
            )
            mask = sam2_wrapper.segment_frame(frame, prompt)
            entry = {"frame": f_path.name}

            if mask is not None:
                out_path = sam2_masks_dir / f"{f_path.name}.png"
                cv2.imwrite(str(out_path), mask)
                sam2_mask_list.append(mask)
                entry["mask_path"] = str(out_path)
                stem = f_path.stem
                if stem in gt_masks:
                    iou = compute_iou(mask, gt_masks[stem])
                    leak = compute_leakage(mask, gt_masks[stem])
                    sam2_ious.append(iou)
                    sam2_leakages.append(leak)
                    entry["iou"] = round(iou, 4)
                    entry["leakage"] = round(leak, 4)
            else:
                sam2_mask_list.append(np.zeros((h, w), dtype=np.uint8))
                entry["error"] = sam2_wrapper.sam2_error_reason

            sam2_per_frame.append(entry)

        sam2_time = time.time() - t0
        sam2_iou = float(np.mean(sam2_ious)) if sam2_ious else 0.0
        sam2_leakage = float(np.mean(sam2_leakages)) if sam2_leakages else 0.0
        sam2_stability = compute_temporal_stability(sam2_mask_list)
        logger.info(
            f"SAM2: {len(frame_files)} frames in {sam2_time:.2f}s, "
            f"IoU={sam2_iou:.4f}, leakage={sam2_leakage:.4f}"
        )
    else:
        logger.warning(
            f"SAM2 not available: "
            f"{sam2_status.get('sam2_error_reason', 'unknown')}. "
            "Legacy-only results."
        )

    # --- Decision ---
    iou_gain = sam2_iou - legacy_iou
    leakage_reduction = legacy_leakage - sam2_leakage

    if not sam2_ran:
        recommendation = (
            "sam2_unavailable — cannot evaluate. Install torch + sam2 "
            "and place checkpoint at SAM2_CHECKPOINT path."
        )
    elif iou_gain >= 0.05 and leakage_reduction > 0:
        if sam2_stability["empty_mask_count"] == 0:
            recommendation = (
                "sam2_improves_segmentation — continue SAM2 development"
            )
        else:
            recommendation = (
                "sam2_partial_improvement — IoU improved but "
                f"{sam2_stability['empty_mask_count']} empty masks detected. "
                "Revisit prompt strategy."
            )
    elif iou_gain < 0.05:
        recommendation = (
            "sam2_no_improvement — do NOT proceed to Depth Anything. "
            "Revisit prompt strategy (center_box vs center_point vs auto, "
            "legacy bbox prompt, manual first-frame prompt)."
        )
    else:
        recommendation = "sam2_mixed — review manually before proceeding"

    results = {
        "sam2_status": sam2_status,
        "legacy_iou": round(legacy_iou, 4),
        "sam2_iou": round(sam2_iou, 4),
        "iou_gain": round(iou_gain, 4),
        "legacy_leakage": round(legacy_leakage, 4),
        "sam2_leakage": round(sam2_leakage, 4),
        "leakage_reduction": round(leakage_reduction, 4),
        "sam2_ran": sam2_ran,
        "runtime_sec": {
            "legacy": round(legacy_time, 2),
            "sam2": round(sam2_time, 2),
        },
        "gpu_available": gpu_available,
        "temporal_stability": {
            "legacy": legacy_stability,
            "sam2": sam2_stability,
        },
        "frames_evaluated": len(frame_files),
        "gt_masks_available": len(gt_masks),
        "gt_frames_matched": list(gt_masks.keys()),
        "final_recommendation": recommendation,
        "legacy_per_frame": legacy_per_frame,
        "sam2_per_frame": sam2_per_frame,
    }

    out_path = output_dir / "sam2_dev_subset_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results written to {out_path}")
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
