import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
import json
import shutil
from enum import Enum

logger = logging.getLogger("texture_frame_filter")

class ProductProfileType(str, Enum):
    BOTTLE = "bottle"
    BOX = "box"
    GENERIC = "generic"

class TextureFrameFilter:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds
        # Defaults
        self.min_sharpness = 20.0
        self.min_luminance = 50.0 # 0-255
        self.max_luminance = 245.0 # avoid blown out
        self.min_mask_coverage = 0.05 # 5% of image
        
    def filter_session_images(
        self, 
        image_folder: Path, 
        output_dir: Path, 
        dense_workspace: Path,
        expected_color: str = "unknown",
        target_count: int = 20,
        product_profile: ProductProfileType = ProductProfileType.GENERIC
    ) -> Dict[str, Any]:
        """
        Analyzes and filters images for texturing.
        Returns path to filtered images directory and metadata.
        """
        # Apply profile-specific thresholds
        if product_profile == ProductProfileType.BOTTLE:
            self.min_mask_coverage = 0.02
        elif product_profile == ProductProfileType.BOX:
            self.min_mask_coverage = 0.10

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
                
        # --- SPRINT v6: Advanced Mask QA & Temporal Filtering ---
        mask_qa_report = {}
        if has_masks:
            logger.info("Running Strict Mask QA & Temporal Filtering...")
            all_mask_stats = {}
            for img_path in images:
                mask_path = mask_folder / (img_path.stem + ".png")
                if not mask_path.exists():
                    mask_path = mask_folder / (img_path.name + ".png")
                if mask_path.exists():
                    all_mask_stats[img_path.name] = self._analyze_mask_quality(mask_path)
            
            if all_mask_stats:
                keys = ["occupancy_ratio", "bbox_area_ratio", "centroid_x", "centroid_y"]
                series = {k: [s[k] for s in all_mask_stats.values()] for k in keys}
                temporal_stats = {
                    k: {"mean": float(np.mean(series[k])), "std": float(np.std(series[k]))} 
                    for k in keys
                }
                for name, s in all_mask_stats.items():
                    for k in keys:
                        std = temporal_stats[k]["std"]
                        s[f"temporal_{k}_zscore"] = abs(s[k] - temporal_stats[k]["mean"]) / std if std > 0.001 else 0.0
                mask_qa_report["temporal_summary"] = temporal_stats
                mask_qa_report["frames"] = all_mask_stats
            
            for s in analyzed_stats:
                if s["name"] in all_mask_stats:
                    s["mask_qa"] = all_mask_stats[s["name"]]
        
        # --- Strict Mask Based Rejection ---
        for s in analyzed_stats:
            mqa = s.get("mask_qa")
            if mqa:
                cur_reasons = s.get("rejection_reasons", [])
                if mqa["occupancy_ratio"] < 0.01:
                    cur_reasons.append("mask_too_small")
                if mqa["border_connected_largest_component"] and mqa["occupancy_ratio"] > 0.2:
                    cur_reasons.append("large_border_touching_component")
                if mqa.get("temporal_occupancy_ratio_zscore", 0) > 3.0:
                    cur_reasons.append("temporal_occupancy_outlier")
                if mqa.get("temporal_centroid_x_zscore", 0) > 3.0 or mqa.get("temporal_centroid_y_zscore", 0) > 3.0:
                    cur_reasons.append("temporal_centroid_jump")
                
                if cur_reasons:
                    s["is_accepted"] = False
                    s["rejection_reasons"] = list(set(cur_reasons))

        # SPRINT Hardening: View-Angle Coverage Based Selection
        selected_stats = []
        coverage_gap_detected = False
        max_gap_degrees = 0.0
        cameras_loaded = False
        azimuths_computed = 0
        selected_azimuths = []

        try:
            from modules.asset_cleanup_pipeline.camera_projection import load_reconstruction_cameras
            
            cameras = load_reconstruction_cameras(dense_workspace)
            if cameras:
                cameras_loaded = True
                name_to_azimuth = {}
                for cam in cameras:
                    R = cam["R"]
                    t = cam["t"]
                    C = -R.T @ t
                    azimuth = float(np.degrees(np.arctan2(C[0], C[2])))
                    name_to_azimuth[cam["name"]] = azimuth
                
                azimuths_computed = len(name_to_azimuth)
                for s in analyzed_stats:
                    s["azimuth"] = name_to_azimuth.get(s["name"])

                # Filter to only those that are accepted AND have azimuths
                coverage_candidates = [s for s in analyzed_stats if s["is_accepted"] and s.get("azimuth") is not None]
                coverage_candidates.sort(key=lambda x: x["azimuth"])

                if coverage_candidates:
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
                            logger.warning(f"Coverage gap of {max_gap_degrees:.1f} degrees detected.")

                    budget = min(target_count, len(coverage_candidates))
                    ideal_angles = np.linspace(-180, 180, budget, endpoint=False)
                    for ideal_angle in ideal_angles:
                        best_match = None
                        min_dist = float("inf")
                        for cand in coverage_candidates:
                            if any(p["name"] == cand["name"] for p in selected_stats): continue
                            dist = abs(cand["azimuth"] - ideal_angle)
                            if dist > 180: dist = 360 - dist
                            quality_adjusted_dist = dist / max(cand.get("ranking_score", 0.1), 0.01)
                            if quality_adjusted_dist < min_dist:
                                min_dist = quality_adjusted_dist
                                best_match = cand
                        if best_match:
                            selected_stats.append(best_match)
                            selected_azimuths.append(best_match["azimuth"])
        except Exception as e:
            logger.error(f"Failed angle-aware selection: {e}")

        if not selected_stats:
            accepted = [s for s in analyzed_stats if s["is_accepted"]]
            accepted.sort(key=lambda x: x.get("ranking_score", 0.0), reverse=True)
            selected_stats = accepted[:target_count]
        
        selected_names = {s["name"] for s in selected_stats}
        for s in analyzed_stats:
            if s["is_accepted"] and s["name"] not in selected_names:
                s["is_accepted"] = False
                s["rejection_reasons"] = ["budget_limit"]
                rejected_stats.append(s)
            elif not s["is_accepted"] and s not in rejected_stats:
                rejected_stats.append(s)

        fallback_used = False
        if not selected_stats and images:
             logger.warning("All images rejected. Fallback to top 5.")
             fallback_used = True
             all_frames = sorted(analyzed_stats + rejected_stats, key=lambda x: x.get("sharpness", 0), reverse=True)
             selected_stats = all_frames[:5]

        for s in selected_stats:
            img_path = Path(s["path"])
            dest_path = selected_dir / img_path.name
            shutil.copy2(img_path, dest_path)
            if has_masks:
                mask_path = mask_folder / (img_path.stem + ".png")
                if not mask_path.exists(): mask_path = mask_folder / (img_path.name + ".png")
                if mask_path.exists():
                    try:
                        img = cv2.imread(str(img_path))
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if img is not None and mask is not None:
                            if mask.shape[:2] != img.shape[:2]:
                                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                            img[mask == 0] = [220, 245, 245]
                            cv2.imwrite(str(masked_dir / img_path.name), img)
                            s["masked_source_generated"] = True
                    except Exception: pass

        # --- v6 Early Gates & Decision Logic ---
        
        # 1. Coverage Gate
        coverage_risk = max_gap_degrees > 90.0
        # Profile-specific thresholds
        gap_threshold = 45.0
        if product_profile == ProductProfileType.BOTTLE:
            gap_threshold = 40.0 # Bottles need strict 360
        elif product_profile == ProductProfileType.BOX:
            gap_threshold = 60.0 # Boxes can have larger gaps between faces
            
        recapture_required = max_gap_degrees > gap_threshold
        recapture_reason = None
        if recapture_required:
            recapture_reason = f"missing_side_coverage (gap: {max_gap_degrees:.1f}° > {gap_threshold}°)"

        # 2. Blur Gate
        avg_sharpness = np.mean([s["sharpness"] for s in selected_stats]) if selected_stats else 0
        if not fallback_used and avg_sharpness < 20.0:
            recapture_required = True
            recapture_reason = "blurry_capture"

        # 3. Mask Quality / SAM2 Recommendation
        try_sam2_masks = False
        bad_mask_count = sum(1 for s in analyzed_stats if s.get("mask_qa", {}).get("occupancy_ratio", 0) < 0.05)
        if bad_mask_count > len(analyzed_stats) * 0.3 and not recapture_required:
            # If coverage is good but masks are bad, recommend SAM2
            if max_gap_degrees < gap_threshold:
                try_sam2_masks = True

        report = {
            "selected_count": len(selected_stats),
            "rejected_count": len(rejected_stats),
            "fallback_used": fallback_used,
            "has_masks_available": has_masks,
            "max_gap_degrees": max_gap_degrees,
            "coverage_risk": coverage_risk,
            "recapture_required": recapture_required,
            "recapture_reason": recapture_reason,
            "try_sam2_masks": try_sam2_masks,
            "selected_frames": selected_stats,
            "mask_qa_report": mask_qa_report,
            "frame_0021_status": next((s for s in analyzed_stats + rejected_stats if s["name"] == "frame_0021.jpg"), {"reason": "not_found"})
        }
        
        with open(output_dir / "selected_texture_frames.json", "w") as f:
            json.dump(report, f, indent=2)
        with open(output_dir / "mask_qa_report.json", "w") as f:
            json.dump(mask_qa_report, f, indent=2)
        with open(output_dir / "rejected_texture_frames.json", "w") as f:
            json.dump(rejected_stats, f, indent=2)
            
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
        
        # 3. Component Analysis
        num_labels, labels, stats_cc, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        component_count = 0
        largest_component_area = 0
        secondary_component_area = 0
        border_connected_largest = False
        
        if num_labels > 1:
            # Sort by area, skipping background (index 0)
            areas = stats_cc[1:, cv2.CC_STAT_AREA]
            sorted_indices = np.argsort(areas)[::-1]
            component_count = len(areas)
            largest_component_area = int(areas[sorted_indices[0]])
            if len(areas) > 1:
                secondary_component_area = int(areas[sorted_indices[1]])
            
            # Check if largest component touches border
            li = sorted_indices[0] + 1
            x, y, w_cc, h_cc = stats_cc[li, :4]
            if x == 0 or y == 0 or (x + w_cc) >= w or (y + h_cc) >= h:
                border_connected_largest = True

        # 4. BBox and Centroid
        coords = np.column_stack(np.where(binary > 0))
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bw, bh = x_max - x_min, y_max - y_min
            bbox_area_ratio = (bw * bh) / (h * w)
            centroid_x = float(np.mean(coords[:, 1])) / w
            centroid_y = float(np.mean(coords[:, 0])) / h
        else:
            bbox_area_ratio = 0.0
            centroid_x = 0.5
            centroid_y = 0.5

        return {
            "occupancy_ratio": float(occupancy),
            "bbox_area_ratio": float(bbox_area_ratio),
            "centroid_x": float(centroid_x),
            "centroid_y": float(centroid_y),
            "border_touch_score": float(border_touch_score),
            "component_count": component_count,
            "largest_component_area_ratio": largest_component_area / (h * w),
            "secondary_component_area_ratio": secondary_component_area / (h * w),
            "border_connected_largest_component": border_connected_largest
        }
