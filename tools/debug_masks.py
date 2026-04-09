import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import sys

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.capture_workflow.object_masker import ObjectMasker
from modules.capture_workflow.quality_analyzer import QualityAnalyzer


def draw_text_block(img: np.ndarray, lines: List[str], x: int, y: int, line_h: int = 22):
    for i, line in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask.copy()


def make_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    red = np.zeros_like(frame)
    red[:, :, 2] = 255

    alpha = 0.35
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(frame, 1.0 - alpha, red, alpha, 0)[mask_bool]
    return overlay


def crop_preview(frame: np.ndarray, bbox: dict, pad_ratio: float = 0.15) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((200, 200, 3), dtype=np.uint8)
    return crop


def build_debug_canvas(
    frame: np.ndarray,
    mask: np.ndarray,
    mask_meta: dict,
    quality_report: dict,
    filename: str,
) -> np.ndarray:
    vis_frame = frame.copy()
    vis_mask = mask_to_bgr(mask)
    overlay = make_overlay(frame, mask)

    bbox = mask_meta.get("bbox")
    centroid = mask_meta.get("centroid")

    if bbox:
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

    if centroid:
        cx, cy = int(centroid["x"]), int(centroid["y"])
        cv2.circle(vis_frame, (cx, cy), 4, (0, 255, 0), -1)
        cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1)

    crop = crop_preview(frame, bbox) if bbox else np.zeros((200, 200, 3), dtype=np.uint8)

    # normalize crop display size
    crop = cv2.resize(crop, (frame.shape[1], frame.shape[0]))

    top_row = np.hstack([vis_frame, vis_mask])
    bottom_row = np.hstack([overlay, crop])
    canvas = np.vstack([top_row, bottom_row])

    # header band
    header_h = 180
    final = np.zeros((canvas.shape[0] + header_h, canvas.shape[1], 3), dtype=np.uint8)
    final[header_h:, :, :] = canvas

    status = "PASS" if quality_report.get("overall_pass", False) else "REJECT"
    status_color = (0, 180, 0) if status == "PASS" else (0, 0, 200)
    cv2.putText(
        final,
        f"{filename} :: {status}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        status_color,
        2,
        cv2.LINE_AA,
    )

    left_lines = [
        f"mask_confidence: {mask_meta.get('mask_confidence', mask_meta.get('confidence', 0.0)):.2f}",
        f"occupancy: {mask_meta.get('occupancy', 0.0):.2%}",
        f"is_clipped: {mask_meta.get('is_clipped', False)}",
        f"fragment_count: {mask_meta.get('fragment_count', 0)}",
        f"largest_contour_ratio: {mask_meta.get('largest_contour_ratio', 0.0):.2f}",
        f"solidity: {mask_meta.get('solidity', 0.0):.2f}",
    ]
    draw_text_block(final, left_lines, 20, 65)

    right_lines = [
        f"blur_score: {quality_report.get('blur_score', 0.0):.2f}",
        f"exposure_score: {quality_report.get('exposure_score', 0.0):.2f}",
        f"center_dist: {quality_report.get('center_dist', 0.0):.2f}",
        f"overall_pass: {quality_report.get('overall_pass', False)}",
        f"failure_reasons: {', '.join(quality_report.get('failure_reasons', [])) or '-'}",
    ]
    draw_text_block(final, right_lines, final.shape[1] // 2, 65)

    # quadrant titles
    title_y = header_h + 25
    cv2.putText(final, "Original + bbox/centroid", (20, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(final, "Binary mask", (frame.shape[1] + 20, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(final, "Overlay", (20, header_h + frame.shape[0] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(final, "Cropped ROI preview", (frame.shape[1] + 20, header_h + frame.shape[0] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return final


def main():
    parser = argparse.ArgumentParser(description="Debug product masks and quality decisions.")
    parser.add_argument("--frames-dir", required=True, help="Directory containing input frames (.jpg/.png).")
    parser.add_argument("--out-dir", required=True, help="Directory where debug images will be written.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of frames to process.")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    if args.limit > 0:
        frame_files = frame_files[: args.limit]

    masker = ObjectMasker()
    analyzer = QualityAnalyzer()

    passed = 0
    failed = 0

    for idx, frame_path in enumerate(frame_files):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        mask, mask_meta = masker.generate_mask(frame)
        report = analyzer.analyze_frame(frame, mask, mask_meta)

        if report["overall_pass"]:
            passed += 1
        else:
            failed += 1

        canvas = build_debug_canvas(frame, mask, mask_meta, report, frame_path.name)
        out_path = out_dir / f"{frame_path.stem}_debug.jpg"
        cv2.imwrite(str(out_path), canvas)

        print(
            f"[{idx+1}/{len(frame_files)}] {frame_path.name} :: "
            f"pass={report['overall_pass']} "
            f"conf={mask_meta.get('mask_confidence', mask_meta.get('confidence', 0.0)):.2f} "
            f"occ={mask_meta.get('occupancy', 0.0):.2%} "
            f"reasons={report.get('failure_reasons', [])}"
        )

    print("\n=== MASK DEBUG SUMMARY ===")
    print(f"frames processed : {len(frame_files)}")
    print(f"pass             : {passed}")
    print(f"reject           : {failed}")
    print(f"output dir       : {out_dir}")


if __name__ == "__main__":
    main()