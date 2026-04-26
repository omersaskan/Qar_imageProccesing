import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import json
import shutil

logger = logging.getLogger("texture_frame_filter")

class TextureFrameFilter:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds
        # Defaults
        self.min_sharpness = 20.0
        self.min_luminance = 50.0 # 0-255
        self.max_luminance = 245.0 # avoid blown out
        self.min_mask_coverage = 0.05 # 5% of image
        
    def filter_session_images(self, image_folder: Path, output_dir: Path, expected_color: str = "unknown") -> Dict[str, Any]:
        """
        Analyzes and filters images for texturing.
        Returns path to filtered images directory and metadata.
        """
        images = sorted(list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png")))
        
        selected_dir = output_dir / "selected_images"
        if selected_dir.exists():
             shutil.rmtree(selected_dir)
        selected_dir.mkdir(parents=True, exist_ok=True)
        
        selected_stats = []
        rejected_stats = []
        
        for img_path in images:
            # Skip known non-RGB patterns
            if any(p in img_path.name.lower() for p in ["mask", "depth", "debug", "normal"]):
                rejected_stats.append({
                    "name": img_path.name,
                    "reason": "non_rgb_pattern",
                    "is_accepted": False
                })
                continue

            stats = self.analyze_frame(img_path, expected_color)
            
            rejection_reasons = self._get_rejection_reasons(stats, expected_color)
            if not rejection_reasons:
                stats["is_accepted"] = True
                selected_stats.append(stats)
                # Copy to filtered dir
                shutil.copy2(img_path, selected_dir / img_path.name)
            else:
                stats["is_accepted"] = False
                stats["rejection_reasons"] = rejection_reasons
                rejected_stats.append(stats)
                
        # Ensure we have at least SOME images. If too restrictive, fallback to best of bad.
        if not selected_stats and images:
             logger.warning("All images rejected by filter. Falling back to top 10 by sharpness.")
             # Simple fallback to prevent complete failure
             all_analyzed = sorted([s for s in rejected_stats if "sharpness" in s], key=lambda x: x["sharpness"], reverse=True)
             for s in all_analyzed[:10]:
                 s["is_accepted"] = True
                 s["fallback"] = True
                 selected_stats.append(s)
                 shutil.copy2(Path(s["path"]), selected_dir / s["name"])

        # Diagnostics
        with open(output_dir / "selected_texture_frames.json", "w") as f:
            json.dump(selected_stats, f, indent=2)
        with open(output_dir / "rejected_texture_frames.json", "w") as f:
            json.dump(rejected_stats, f, indent=2)
            
        # Contact sheet of selected
        self._generate_contact_sheet(selected_stats, output_dir / "selected_texture_frames_contact_sheet.png")

        return {
            "selected_count": len(selected_stats),
            "rejected_count": len(rejected_stats),
            "selected_images_dir": selected_dir,
            "selected_frames": selected_stats
        }

    def analyze_frame(self, path: Path, expected_color: str) -> Dict[str, Any]:
        img = cv2.imread(str(path))
        if img is None:
            return {"name": path.name, "path": str(path), "error": "load_failed"}
            
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Brightness/Luminance
        avg_luminance = np.mean(gray)
        
        # 3. Subject Detection (Heuristic if no mask)
        # We look for non-neutral pixels or edges in the center
        # For now, let's use a simple saliency or center-weighted variance
        center_h, center_w = h // 4, w // 4
        center_roi = gray[center_h:3*center_h, center_w:3*center_w]
        center_variance = np.var(center_roi)
        
        # 4. Clipping check
        # Check if high contrast edges hit the boundary
        edges = cv2.Canny(gray, 50, 150)
        edge_sum = np.sum(edges[0, :]) + np.sum(edges[-1, :]) + np.sum(edges[:, 0]) + np.sum(edges[:, -1])
        clipping_score = edge_sum / (255 * (2*h + 2*w))

        # 5. Color match if white_cream
        color_match = 1.0
        if expected_color == "white_cream":
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            s = hsv[:,:,1]
            v = hsv[:,:,2]
            # White/Cream should be high value, low saturation
            white_mask = (v > 180) & (s < 50)
            color_match = np.count_nonzero(white_mask) / (h * w)

        return {
            "name": path.name,
            "path": str(path),
            "sharpness": float(sharpness),
            "luminance": float(avg_luminance),
            "center_variance": float(center_variance),
            "clipping_score": float(clipping_score),
            "color_match_score": float(color_match),
            "resolution": [w, h]
        }

    def _get_rejection_reasons(self, stats: Dict[str, Any], expected_color: str) -> List[str]:
        reasons = []
        if "error" in stats: return ["load_failed"]
        
        if stats["sharpness"] < self.min_sharpness:
            reasons.append(f"too_blurry({stats['sharpness']:.1f})")
        if stats["luminance"] < self.min_luminance:
            reasons.append(f"too_dark({stats['luminance']:.1f})")
        if stats["luminance"] > self.max_luminance:
            reasons.append(f"too_bright({stats['luminance']:.1f})")
        if stats["clipping_score"] > 0.1:
            reasons.append(f"subject_clipped({stats['clipping_score']:.2f})")
            
        if expected_color == "white_cream" and stats["color_match_score"] < 0.1:
            reasons.append(f"color_mismatch_white_cream({stats['color_match_score']:.2f})")
            
        return reasons

    def _generate_contact_sheet(self, selected_stats: List[Dict[str, Any]], output_path: Path):
        if not selected_stats: return
        
        thumbnails = []
        # Limit to 16 best frames for the sheet
        top_frames = sorted(selected_stats, key=lambda x: x["sharpness"], reverse=True)[:16]
        
        for s in top_frames:
            img = cv2.imread(s["path"])
            if img is not None:
                thumbnails.append(cv2.resize(img, (256, 256)))
                
        if not thumbnails: return
        
        grid_size = int(np.ceil(np.sqrt(len(thumbnails))))
        h, w = 256, 256
        sheet = np.zeros((grid_size * h, grid_size * w, 3), dtype=np.uint8)
        
        for idx, thumb in enumerate(thumbnails):
            r, c = divmod(idx, grid_size)
            sheet[r*h:(r+1)*h, c*w:(c+1)*w] = thumb
            
        cv2.imwrite(str(output_path), sheet)
