import cv2
import numpy as np
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.getcwd())

from modules.capture_workflow.object_masker import ObjectMasker
from modules.capture_workflow.quality_analyzer import QualityAnalyzer

def debug_masks(images_dir: str, output_dir: str):
    masker = ObjectMasker()
    analyzer = QualityAnalyzer()
    
    img_path = Path(images_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    files = list(img_path.glob("*.jpg")) + list(img_path.glob("*.png"))
    print(f"DEBUGGING {len(files)} masks...")
    
    for f in files:
        frame = cv2.imread(str(f))
        if frame is None: continue
        
        mask, meta = masker.generate_mask(frame)
        analysis = analyzer.analyze_frame(frame, mask)
        
        # Create composite debug image
        h, w = frame.shape[:2]
        canvas = np.zeros((h, w*2, 3), np.uint8)
        canvas[:, :w] = frame
        
        # Colorize mask for better visibility
        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        color_mask[mask > 0] = [0, 255, 0] # Green for foreground
        
        # Blend mask with frame on the right side
        blended = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
        canvas[:, w:] = blended
        
        # Draw BBox and Centroid
        bx, by, bw, bh = meta.get("bbox", (0,0,0,0))
        cv2.rectangle(canvas, (w + bx, by), (w + bx + bw, by + bh), (0, 0, 255), 2)
        
        cx, cy = meta.get("centroid", (w//2, h//2))
        cv2.circle(canvas, (w + cx, cy), 5, (255, 0, 0), -1)
        
        # Add labels
        y_off = 30
        labels = [
            f"CONF: {meta['confidence']:.2f}",
            f"OCCU: {meta['occupancy']:.2%}",
            f"FRAG: {meta['fragment_count']}",
            f"SOLID: {meta['solidity']:.2f}",
            f"LARGEST: {meta['largest_contour_ratio']:.2f}",
            f"PASS: {analysis['overall_pass']}"
        ]
        
        if analysis["failure_reasons"]:
            labels.append("REASONS:")
            labels.extend(analysis["failure_reasons"])
            
        for label in labels:
            cv2.putText(canvas, label, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_off += 30
            
        cv2.imwrite(str(out_path / f.name), canvas)
        print(f"  Processed {f.name}: pass={analysis['overall_pass']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/debug/masks")
    args = parser.parse_args()
    
    debug_masks(args.input, args.output)
