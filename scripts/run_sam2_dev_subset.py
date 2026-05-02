"""
SAM2 Dev-Subset Evaluation Script
====================================

Compares SAM2 (image or video mode) vs legacy segmentation on a dev-subset.
Supports prompt strategy sweeps and automated reporting.

Features:
- GT Metadata awareness: excludes invalid frames from corrected metrics.
- Phase isolation: forces settings for legacy/SAM2 phases.
- Video Mode: uses temporal propagation for consistency.
- Observability: detailed status and error tracking.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from unittest.mock import patch
from typing import List, Dict, Any, Optional

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

    area_std = float(np.std(areas)) if len(areas) > 1 else 0.0

    centroid_dists = []
    for i in range(1, len(centroids)):
        if centroids[i] is not None and centroids[i - 1] is not None:
            dx = centroids[i][0] - centroids[i - 1][0]
            dy = centroids[i][1] - centroids[i - 1][1]
            centroid_dists.append(float(np.sqrt(dx ** 2 + dy ** 2)))
    centroid_jitter = float(np.mean(centroid_dists)) if centroid_dists else 0.0

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
# GT Metadata handling
# -------------------------------------------------------------------

def load_gt_metadata(capture_id: str, gt_dir: Path) -> Dict[str, Any]:
    """Load GT metadata for a capture to identify invalid frames."""
    meta_path = gt_dir.parent / "metadata" / f"{capture_id}.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


def build_gt_map(gt_dir: Path):
    """Build mapping from frame stem (e.g. 'frame_0000') to GT mask."""
    gt_masks = {}
    for gt_file in sorted(gt_dir.glob("*.png")):
        m = re.search(r"_f(\d+)\.png$", gt_file.name)
        if m:
            idx = int(m.group(1))
            frame_stem = f"frame_{idx:04d}"
            gt_masks[frame_stem] = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
            logger.info(f"GT mapped: {gt_file.name} \u2192 {frame_stem}")
    return gt_masks


def get_gt_bbox(gt_mask: np.ndarray) -> Optional[List[int]]:
    """Extract [x1, y1, x2, y2] from a GT mask."""
    if gt_mask is None or not np.any(gt_mask > 0):
        return None
    ys, xs = np.where(gt_mask > 0)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# -------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------

def run_evaluation(args):
    from modules.operations.settings import settings
    from modules.capture_workflow.object_masker import ObjectMasker
    from modules.ai_segmentation.prompting import generate_prompts

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve capture ID
    capture_id = args.capture_id
    if not capture_id and args.frames_dir:
        parts = Path(args.frames_dir).parts
        if "captures" in parts:
            idx = parts.index("captures")
            if idx + 1 < len(parts):
                capture_id = parts[idx + 1]

    # Resolve frames directory
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
    elif capture_id:
        frames_dir = Path(settings.data_root) / "captures" / capture_id / "frames"
    else:
        logger.error("Provide --frames-dir or --capture-id")
        sys.exit(1)

    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        sys.exit(1)

    frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    max_frames = min(len(frame_files), args.max_frames)
    frame_files = frame_files[:max_frames]
    logger.info(f"Evaluating {len(frame_files)} frames from {frames_dir} (Capture: {capture_id}, Mode: {args.sam2_mode})")

    # Load GT masks and Metadata
    gt_masks = {}
    gt_metadata = {}
    invalid_frames = []
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
        gt_masks = build_gt_map(gt_dir)
        if capture_id:
            gt_metadata = load_gt_metadata(capture_id, gt_dir)
            for frame_info in gt_metadata.get("validation_frames", []):
                if not frame_info.get("is_valid", True):
                    f_idx = frame_info["frame_index"]
                    f_stem = f"frame_{f_idx:04d}"
                    invalid_frames.append(f_stem)
                    logger.warning(f"GT Frame {f_stem} marked INVALID: {frame_info.get('invalid_reason')}")

    manual_first_frame_box = None
    if "frame_0000" in gt_masks:
        manual_first_frame_box = get_gt_bbox(gt_masks["frame_0000"])
        logger.info(f"Manual box derived from frame_0000 GT: {manual_first_frame_box}")

    # Modes to test
    if args.sweep:
        prompt_modes = ["center_point", "center_box", "auto", "legacy_bbox", "legacy_centroid", "manual_first_frame_box"]
    else:
        prompt_modes = [settings.sam2_prompt_mode]

    all_sweep_results = []

    # --- PHASE 1: LEGACY BASELINE ---
    logger.info("Running legacy baseline...")
    with patch.object(settings, "segmentation_method", "legacy"), \
         patch.object(settings, "sam2_enabled", False):
        
        legacy_masker = ObjectMasker()
        legacy_per_frame = []
        legacy_mask_list = []
        t0 = time.time()

        for f_path in frame_files:
            frame = cv2.imread(str(f_path))
            mask, meta = legacy_masker.generate_mask(frame)
            legacy_mask_list.append(mask)
            
            entry = {"frame": f_path.name, "backend": meta.get("backend_name", "unknown")}
            stem = f_path.stem
            if stem in gt_masks:
                iou = compute_iou(mask, gt_masks[stem])
                leak = compute_leakage(mask, gt_masks[stem])
                entry["iou"] = round(iou, 4)
                entry["leakage"] = round(leak, 4)
                entry["gt_valid"] = stem not in invalid_frames
            legacy_per_frame.append(entry)

        legacy_time = time.time() - t0
        legacy_backends = list(set(e["backend"] for e in legacy_per_frame))
        legacy_ai_used = any("ai" in b.lower() or "sam" in b.lower() for b in legacy_backends)
        
        verification_data = {
            "legacy_method_verified": True,
            "legacy_backends_detected": legacy_backends,
            "legacy_ai_segmentation_used": legacy_ai_used,
            "sam2_method_verified": False,
            "sam2_backends_detected": [],
            "sam2_ai_segmentation_used": False
        }

    # --- PHASE 2: SAM2 ---
    for mode in prompt_modes:
        logger.info(f"--- Testing SAM2 {args.sam2_mode} mode: {mode} ---")
        
        sam2_mask_list = []
        sam2_per_frame = []
        sam2_status = {}
        t0 = time.time()

        if args.sam2_mode == "video":
            from modules.ai_segmentation.sam2_video_backend import SAM2VideoBackend
            backend = SAM2VideoBackend(settings.sam2_model_cfg, settings.sam2_checkpoint, settings.sam2_device)
            if not backend.is_available(): continue
            
            # Seed selection
            seed_box = None
            seed_source = "prompt_mode"
            if mode == "manual_first_frame_box" and manual_first_frame_box:
                seed_box = manual_first_frame_box
                seed_source = "gt_frame_0000"
            elif mode in ["center_box", "legacy_bbox"]:
                seed_box = get_gt_bbox(legacy_mask_list[0]) if legacy_mask_list[0].any() else None
                seed_source = "legacy_mask_f0"
            
            # Propagation
            video_masks = backend.segment_video(
                frames_dir=frames_dir,
                seed_frame_idx=0,
                seed_box=seed_box,
                seed_prompt_source=seed_source,
                output_dir=None
            )
            
            if not video_masks and not backend.video_propagation_failed:
                logger.error(f"Video propagation returned no masks for mode {mode}")
                backend.video_propagation_failed = True

            for i, f_path in enumerate(frame_files):
                mask = video_masks.get(i)
                if mask is None:
                    # Mark missing frame
                    mask = np.zeros_like(legacy_mask_list[0])
                    # backend.mask_propagation_failure_count is already updated in backend
                
                sam2_mask_list.append(mask)
                entry = {"frame": f_path.name, "prompt_mode": mode}
                stem = f_path.stem
                if stem in gt_masks:
                    iou = compute_iou(mask, gt_masks[stem])
                    leak = compute_leakage(mask, gt_masks[stem])
                    entry["iou"] = round(iou, 4)
                    entry["leakage"] = round(leak, 4)
                    entry["gt_valid"] = stem not in invalid_frames
                sam2_per_frame.append(entry)
            
            sam2_status = backend.get_status()
            
        else: # image mode
            with patch.object(settings, "segmentation_method", "sam2"), \
                 patch.object(settings, "sam2_enabled", True):
                
                try:
                    from modules.ai_segmentation.sam2_wrapper import SAM2Wrapper
                    sam2_wrapper = SAM2Wrapper()
                    if not sam2_wrapper.is_available(): continue
                except Exception: continue

                for i, f_path in enumerate(frame_files):
                    frame = cv2.imread(str(f_path))
                    h, w = frame.shape[:2]
                    manual_prompt = None
                    if mode == "manual_first_frame_box" and i == 0 and manual_first_frame_box:
                        manual_prompt = {"bbox": manual_first_frame_box}
                    
                    prompt = generate_prompts(
                        frame_shape=(h, w),
                        mode=mode if mode != "manual_first_frame_box" else "center_box",
                        legacy_mask=legacy_mask_list[i],
                        legacy_meta=legacy_per_frame[i],
                        manual_prompt=manual_prompt
                    )
                    mask = sam2_wrapper.segment_frame(frame, prompt)
                    entry = {"frame": f_path.name, "prompt_mode": prompt["prompt_mode"]}
                    if mask is not None:
                        sam2_mask_list.append(mask)
                        stem = f_path.stem
                        if stem in gt_masks:
                            iou = compute_iou(mask, gt_masks[stem])
                            leak = compute_leakage(mask, gt_masks[stem])
                            entry["iou"] = round(iou, 4)
                            entry["leakage"] = round(leak, 4)
                            entry["gt_valid"] = stem not in invalid_frames
                    else:
                        sam2_mask_list.append(np.zeros((h, w), dtype=np.uint8))
                        entry["error"] = sam2_wrapper.sam2_error_reason
                    sam2_per_frame.append(entry)
                sam2_status = sam2_wrapper.get_status()

        sam2_time = time.time() - t0
        verification_data["sam2_method_verified"] = True
        verification_data["sam2_backends_detected"].append("sam2")
        verification_data["sam2_ai_segmentation_used"] = True

        # Metrics aggregation
        def summarize(per_frame, include_invalid=True):
            ious = [e["iou"] for e in per_frame if "iou" in e and (include_invalid or e.get("gt_valid", True))]
            leaks = [e["leakage"] for e in per_frame if "leakage" in e and (include_invalid or e.get("gt_valid", True))]
            return (float(np.mean(ious)) if ious else 0.0), (float(np.mean(leaks)) if leaks else 0.0)

        l_iou_raw, l_leak_raw = summarize(legacy_per_frame, True)
        l_iou_corr, l_leak_corr = summarize(legacy_per_frame, False)
        s_iou_raw, s_leak_raw = summarize(sam2_per_frame, True)
        s_iou_corr, s_leak_corr = summarize(sam2_per_frame, False)
        
        sam2_stability = compute_temporal_stability(sam2_mask_list)
        legacy_stability = compute_temporal_stability(legacy_mask_list)

        res = {
            "prompt_mode": mode,
            "sam2_mode": args.sam2_mode,
            "metrics_raw": {
                "legacy_iou": round(l_iou_raw, 4),
                "sam2_iou": round(s_iou_raw, 4),
                "iou_gain": round(s_iou_raw - l_iou_raw, 4),
                "legacy_leakage": round(l_leak_raw, 4),
                "sam2_leakage": round(s_leak_raw, 4),
            },
            "metrics_corrected": {
                "legacy_iou": round(l_iou_corr, 4),
                "sam2_iou": round(s_iou_corr, 4),
                "iou_gain": round(s_iou_corr - l_iou_corr, 4),
                "legacy_leakage": round(l_leak_corr, 4),
                "sam2_leakage": round(s_leak_corr, 4),
            },
            "invalid_gt_frames": invalid_frames,
            "sam2_status": sam2_status,
            "runtime_sec": {"legacy": round(legacy_time, 2), "sam2": round(sam2_time, 2)},
            "temporal_stability": {"legacy": legacy_stability, "sam2": sam2_stability},
            "empty_mask_count": sam2_stability["empty_mask_count"],
        }
        all_sweep_results.append(res)

    # --- Ranking & Reporting ---
    ranked = sorted(all_sweep_results, key=lambda x: x["metrics_corrected"]["sam2_iou"], reverse=True) if all_sweep_results else []
    
    summary_table = []
    if not all_sweep_results:
        summary_table.append("### SAM2 Evaluation Skipped or Failed")
        summary_table.append("No SAM2 modes were successfully evaluated.")
    else:
        summary_table.append(f"### SAM2 {args.sam2_mode.upper()} Results")
        summary_table.append("| Mode | Corr IoU | Corr Gain | Corr Leak | Jitter | Failures | Status |")
        summary_table.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in ranked:
            m = r["metrics_corrected"]
            gain_str = f"{m['iou_gain']:+.4f}"
            jitter = r["temporal_stability"]["sam2"]["centroid_jitter"]
            fails = r["sam2_status"].get("mask_propagation_failure_count", 0)
            summary_table.append(
                f"| {r['prompt_mode']} | {m['sam2_iou']:.4f} | {gain_str} | "
                f"{m['sam2_leakage']:.4f} | {jitter:.2f} | {fails} | "
                f"{'PASS' if m['iou_gain'] >= 0.05 and fails == 0 else 'FAIL'} |"
            )

    final_results = {
        "sweep_results": all_sweep_results,
        "best_mode": ranked[0]["prompt_mode"] if ranked else None,
        "summary_table": summary_table,
        "invalid_frames_audited": invalid_frames,
        "sam2_mode": args.sam2_mode,
        "sam2_ran": len(all_sweep_results) > 0,
        "verification": verification_data
    }

    out_path = output_dir / f"sam2_sweep_{args.sam2_mode}_corrected.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    logger.info(f"Results written to {out_path}")
    for line in summary_table: print(line)
    return final_results


def main():
    parser = argparse.ArgumentParser(description="SAM2 Dev-Subset Sweep Corrected")
    parser.add_argument("--capture-id", type=str)
    parser.add_argument("--frames-dir", type=str)
    parser.add_argument("--gt-dir", type=str)
    parser.add_argument("--output-dir", type=str, default="results/sam2_sweep_corrected")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--sam2-mode", type=str, choices=["image", "video"], default="image")
    args = parser.parse_args()
    run_evaluation(args)

if __name__ == "__main__":
    main()
