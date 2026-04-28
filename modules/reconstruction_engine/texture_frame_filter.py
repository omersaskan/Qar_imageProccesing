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
        
    def filter_session_images(self, image_folder: Path, output_dir: Path, expected_color: str = "unknown", target_count: int = 20) -> Dict[str, Any]:
        """
        Analyzes and filters images for texturing.
        Returns path to filtered images directory and metadata.
        """
        images = sorted(list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png")))
        
        selected_dir = output_dir / "selected_images"
        if selected_dir.exists():
             shutil.rmtree(selected_dir)
        selected_dir.mkdir(parents=True, exist_ok=True)
        
        # SPRINT 5C: Real Mask Resolver - Robust Evaluation
        masked_dir = output_dir / "selected_images_masked"
        if masked_dir.exists():
            shutil.rmtree(masked_dir)
        masked_dir.mkdir(parents=True, exist_ok=True)
        
        potential_mask_paths = [
            image_folder.parent / "stereo" / "masks",
            image_folder.parent / "masks",
            image_folder.parent.parent / "dense" / "stereo" / "masks",
            image_folder.parent.parent / "dense" / "masks",
            image_folder.parent.parent / "masks",
            image_folder / "masks", # Some workflows put masks inside images folder
        ]
        
        best_mask_folder = None
        best_match_score = -1
        
        # We'll use a sample image to check dimensions
        sample_img = None
        if images:
            sample_img = cv2.imread(str(images[0]))
        
        for p in potential_mask_paths:
            exists = p.exists()
            mask_files = list(p.glob("*.png")) + list(p.glob("*.jpg"))
            mask_count = len(mask_files)
            
            if not exists or mask_count == 0:
                logger.debug(f"Mask candidate: {p} - exists={exists}, count={mask_count}")
                continue
                
            # Match evaluation
            stem_matches = 0
            dim_matches = 0
            
            image_stems = {img.stem for img in images}
            image_names = {img.name for img in images}
            for mask_path in mask_files[:20]: # Sample first 20 for speed
                # Try direct stem match OR stem-without-extension match
                if mask_path.stem in image_stems or mask_path.stem in image_names:
                    stem_matches += 1
                    # Check dimension if it matches stem
                    if sample_img is not None:
                         m_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                         if m_img is not None and m_img.shape[:2] == sample_img.shape[:2]:
                             dim_matches += 1
            
            # Normalize scores
            sample_size = min(20, mask_count)
            match_ratio = stem_matches / sample_size if sample_size > 0 else 0
            
            logger.info(f"Mask candidate: {p} | exists=True | count={mask_count} | stem_matches={stem_matches}/{sample_size} | dim_matches={dim_matches}/{sample_size}")
            
            # Selection criteria: high stem match and some dimension matches
            # Add a small boost for dimension matches to break ties
            dim_ratio = dim_matches / sample_size if sample_size > 0 else 0
            final_score = match_ratio + (dim_ratio * 0.1)
            
            if match_ratio > 0.5 and final_score > best_match_score:
                best_match_score = final_score
                best_mask_folder = p
        
        mask_folder = best_mask_folder
        has_masks = mask_folder is not None
        mask_count = len(list(mask_folder.glob("*"))) if has_masks else 0
        logger.info(f"Selected mask_folder: {mask_folder} (score={best_match_score}, count={mask_count})")

        analyzed_stats = []
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
                # Calculate ranking score
                # Normalized components: 
                # sharpness (capped at 500), luminance (centered at 128), color_match (0-1)
                s_norm = min(stats["sharpness"] / 500.0, 1.0)
                l_norm = 1.0 - abs(stats["luminance"] - 128) / 128.0
                c_norm = stats["color_match_score"]
                
                stats["ranking_score"] = (s_norm * 0.4) + (l_norm * 0.2) + (c_norm * 0.4)
                analyzed_stats.append(stats)
            else:
                stats["is_accepted"] = False
                stats["rejection_reasons"] = rejection_reasons
                rejected_stats.append(stats)
                
        # SPRINT Hardening: View-Angle Coverage Based Selection
        selected_stats = []
        coverage_gap_detected = False
        max_gap_degrees = 0.0

        try:
            from modules.asset_cleanup_pipeline.camera_projection import load_reconstruction_cameras
            # Try to load cameras from the same job structure
            # output_dir is likely the job_dir or a subfolder of it
            recon_root = output_dir
            if not (recon_root / "sparse").exists() and not (recon_root / "dense").exists():
                # Try parent if we are in a subfolder
                recon_root = output_dir.parent
            
            cameras = load_reconstruction_cameras(recon_root)
            if cameras:
                # Calculate azimuths for all analyzed frames
                name_to_azimuth = {}
                for cam in cameras:
                    R = cam["R"]
                    t = cam["t"]
                    C = -R.T @ t
                    azimuth = float(np.degrees(np.arctan2(C[0], C[2])))
                    name_to_azimuth[cam["name"]] = azimuth
                
                for s in analyzed_stats:
                    s["azimuth"] = name_to_azimuth.get(s["name"])

                # Filter to only those with azimuths
                coverage_candidates = [s for s in analyzed_stats if s.get("azimuth") is not None]
                coverage_candidates.sort(key=lambda x: x["azimuth"])

                if coverage_candidates:
                    # 1. Analyze Gaps
                    gaps = []
                    for i in range(len(coverage_candidates)):
                        next_idx = (i + 1) % len(coverage_candidates)
                        diff = coverage_candidates[next_idx]["azimuth"] - coverage_candidates[i]["azimuth"]
                        if diff < 0: diff += 360
                        gaps.append(diff)
                    
                    if gaps:
                        max_gap_degrees = max(gaps)
                        if max_gap_degrees > 45.0:
                            coverage_gap_detected = True
                            logger.warning(f"Coverage gap of {max_gap_degrees:.1f} degrees detected in reconstruction.")

                    # 2. Angle-Aware Selection
                    # Limit n to budget
                    budget = min(target_count, len(coverage_candidates))
                    
                    # Selection strategy: spread frames evenly across azimuths
                    # We pick frames that are as close as possible to ideal spread points
                    ideal_angles = np.linspace(-180, 180, budget, endpoint=False)
                    for ideal_angle in ideal_angles:
                        # Find closest candidate that hasn't been picked yet
                        best_match = None
                        min_dist = float("inf")
                        for cand in coverage_candidates:
                            if any(p["name"] == cand["name"] for p in selected_stats):
                                continue
                            
                            dist = abs(cand["azimuth"] - ideal_angle)
                            if dist > 180: dist = 360 - dist
                            
                            # Weight by ranking_score to prefer higher quality near that angle
                            quality_adjusted_dist = dist / max(cand.get("ranking_score", 0.1), 0.01)
                            
                            if quality_adjusted_dist < min_dist:
                                min_dist = quality_adjusted_dist
                                best_match = cand
                        
                        if best_match:
                            selected_stats.append(best_match)
                else:
                    logger.warning("No camera azimuths available for analyzed frames. Falling back to quality ranking.")
            else:
                logger.warning("No reconstruction cameras found. Falling back to quality ranking.")
        except Exception as e:
            logger.error(f"Failed angle-aware selection: {e}. Falling back to quality ranking.")

        # Fallback to simple top-N if angle selection failed or was skipped
        if not selected_stats:
            selected_stats = analyzed_stats[:target_count]
        
        # If we selected some, the rest of analyzed are effectively rejected for "budget"
        selected_names = {s["name"] for s in selected_stats}
        for s in analyzed_stats:
            if s["name"] not in selected_names:
                s["is_accepted"] = False
                s["rejection_reasons"] = ["below_top_n_threshold_or_angle_coverage"]
                rejected_stats.append(s)

        # Ensure we have at least SOME images. If too restrictive, fallback to best of bad.
        fallback_used = False
        if not selected_stats and images:
             logger.warning("All images rejected by filter. Falling back to top 10 by sharpness.")
             fallback_used = True
             all_analyzed = sorted([s for s in rejected_stats if "sharpness" in s], key=lambda x: x["sharpness"], reverse=True)
             for s in all_analyzed[:10]:
                 s["is_accepted"] = True
                 s["fallback"] = True
                 selected_stats.append(s)

        # Final copy and masking
        for s in selected_stats:
            img_path = Path(s["path"])
            dest_path = selected_dir / img_path.name
            shutil.copy2(img_path, dest_path)
            
            # Masking logic
            if has_masks:
                # Robust mask discovery: try stem.png and name.png
                mask_path = mask_folder / (img_path.stem + ".png")
                if not mask_path.exists():
                    mask_path = mask_folder / (img_path.name + ".png")
                
                if mask_path.exists():
                    mask_stats = self._analyze_mask_quality(mask_path)
                    s["mask_qa"] = mask_stats
                    
                    # SPRINT Hardening: Reject if mask is terrible
                    if mask_stats["occupancy_ratio"] < 0.01 or mask_stats["border_touch_score"] > 0.8:
                        logger.warning(f"Rejecting masked frame {img_path.name} due to poor mask QA: {mask_stats}")
                        s["masked_source_generated"] = False
                        continue

                    try:
                        img = cv2.imread(str(img_path))
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if img is not None and mask is not None:
                            # Resize mask if needed
                            if mask.shape[:2] != img.shape[:2]:
                                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                            
                            # Neutralize background to white/cream (approx #F5F5DC)
                            # Using BGR: (220, 245, 245) for cream
                            img[mask == 0] = [220, 245, 245]
                            cv2.imwrite(str(masked_dir / img_path.name), img)
                            s["masked_source_generated"] = True
                        else:
                            s["masked_source_generated"] = False
                    except Exception as e:
                        logger.warning(f"Failed to generate masked image for {img_path.name}: {e}")
                        s["masked_source_generated"] = False
                else:
                    s["masked_source_generated"] = False

        # Diagnostics
        report = {
            "selected_count": len(selected_stats),
            "rejected_count": len(rejected_stats),
            "fallback_used": fallback_used,
            "has_masks_available": has_masks,
            "coverage_gap_detected": coverage_gap_detected,
            "max_gap_degrees": max_gap_degrees,
            "recapture_required": coverage_gap_detected,
            "recapture_reason": "missing side coverage" if coverage_gap_detected else None,
            "selected_images_dir": str(selected_dir),
            "masked_images_dir": str(masked_dir) if has_masks else None,
            "selected_frames": selected_stats,
            "rejected_frames": rejected_stats
        }
        with open(output_dir / "selected_texture_frames.json", "w") as f:
            json.dump(report, f, indent=2)
        with open(output_dir / "rejected_texture_frames.json", "w") as f:
            json.dump(rejected_stats, f, indent=2)
            
        # Contact sheet of selected
        self._generate_contact_sheet(selected_stats, output_dir / "selected_texture_frames_contact_sheet.png")

        return report

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
            s_chan = hsv[:,:,1]
            v_chan = hsv[:,:,2]
            # White/Cream should be high value, low saturation
            white_mask = (v_chan > 180) & (s_chan < 50)
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
        
        # Stricter thresholds for SPRINT 5C
        if stats["sharpness"] < 25.0:
            reasons.append(f"too_blurry({stats['sharpness']:.1f})")
        if stats["luminance"] < 40.0:
            reasons.append(f"too_dark({stats['luminance']:.1f})")
        if stats["luminance"] > 250.0:
            reasons.append(f"too_bright({stats['luminance']:.1f})")
        if stats["clipping_score"] > 0.15:
            reasons.append(f"subject_clipped({stats['clipping_score']:.2f})")
            
        if expected_color == "white_cream" and stats["color_match_score"] < 0.22:
            reasons.append(f"color_mismatch_white_cream({stats['color_match_score']:.2f})")
            
        return reasons

    def _generate_contact_sheet(self, selected_stats: List[Dict[str, Any]], output_path: Path):
        if not selected_stats: return
        
        thumbnails = []
        # Limit to 16 best frames for the sheet
        top_frames = sorted(selected_stats, key=lambda x: x.get("ranking_score", 0.0), reverse=True)[:16]
        
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

    def _analyze_mask_quality(self, mask_path: Path) -> Dict[str, Any]:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return {"occupancy_ratio": 0.0, "border_touch_score": 1.0, "error": "load_failed"}
        
        h, w = mask.shape
        binary = (mask > 127).astype(np.uint8)
        
        # 1. Occupancy
        white_pixels = np.count_nonzero(binary)
        occupancy = white_pixels / (h * w)
        
        # 2. Border Touch
        top = np.any(binary[0, :])
        bottom = np.any(binary[-1, :])
        left = np.any(binary[:, 0])
        right = np.any(binary[:, -1])
        border_touch_score = (int(top) + int(bottom) + int(left) + int(right)) / 4.0
        
        # 3. Foreground BBox Quality
        coords = np.column_stack(np.where(binary > 0))
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bw, bh = x_max - x_min, y_max - y_min
            bbox_occupancy = white_pixels / max(bw * bh, 1)
            # Centrality
            cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
            centrality = 1.0 - (abs(cx - w/2) / (w/2) + abs(cy - h/2) / (h/2)) / 2.0
        else:
            bbox_occupancy = 0.0
            centrality = 0.0
            
        return {
            "occupancy_ratio": float(occupancy),
            "border_touch_score": float(border_touch_score),
            "bbox_occupancy": float(bbox_occupancy),
            "centrality": float(centrality)
        }
