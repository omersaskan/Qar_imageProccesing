# modules/capture_workflow/frame_extractor.py
from __future__ import annotations

import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    QualityThresholds,
    ExtractionConfig,
    SegmentationConfig,
    default_quality_thresholds,
    default_extraction_config,
    default_segmentation_config,
)
from .quality_analyzer import QualityAnalyzer
from .object_masker import ObjectMasker
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("extractor")


class FrameExtractor:
    def __init__(
        self,
        quality_analyzer: Optional[QualityAnalyzer] = None,
        object_masker: Optional[ObjectMasker] = None,
        thresholds: Optional[QualityThresholds] = None,
        config: Optional[ExtractionConfig] = None,
        seg_config: Optional[SegmentationConfig] = None,
    ):
        self.thresholds = (thresholds or default_quality_thresholds).model_copy(deep=True)
        self.config = (config or default_extraction_config).model_copy(deep=True)
        self.seg_config = (seg_config or default_segmentation_config).model_copy(deep=True)
        self.quality_analyzer = quality_analyzer or QualityAnalyzer(self.thresholds)
        self.object_masker = object_masker or ObjectMasker(
            thresholds=self.thresholds,
            config=self.seg_config,
        )

    def _apply_object_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        expanded_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        return cv2.bitwise_and(frame, frame, mask=expanded_mask)

    def _prepare_object_centric_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        bbox: Optional[Dict[str, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Suppress scene pixels outside an expanded object ROI without crop/resize.

        Important:
        - This method is now used only for masked preview/debug output.
        - The reconstruction input .jpg remains the raw frame, because COLMAP can
          need background/table features for stable camera pose estimation.
        """
        if bbox is None:
            return self._apply_object_mask(frame, mask), mask
        if self.config.roi_mode != "mask_suppression":
            raise ValueError(f"Unsupported ROI mode: {self.config.roi_mode}")

        h, w = mask.shape[:2]
        pad_x = max(8, int(bbox["w"] * self.config.roi_pad_x_ratio))
        pad_y = max(8, int(bbox["h"] * self.config.roi_pad_y_ratio))

        x1 = max(0, bbox["x"] - pad_x)
        y1 = max(0, bbox["y"] - pad_y)
        x2 = min(w, bbox["x"] + bbox["w"] + pad_x)
        y2 = min(h, bbox["y"] + bbox["h"] + pad_y)

        focus_mask = np.zeros_like(mask)
        focus_mask[y1:y2, x1:x2] = 255
        refined_mask = cv2.bitwise_and(mask, focus_mask)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        original_area = float(np.sum(mask > 0))
        refined_area = float(np.sum(refined_mask > 0))
        if original_area > 0 and refined_area < original_area * self.config.roi_min_retained_area_ratio:
            refined_mask = mask

        focused_frame = self._apply_object_mask(frame, refined_mask)
        return focused_frame, refined_mask

    def _write_verified_image(
        self,
        image_path: Path,
        image: np.ndarray,
        label: str,
        read_flag: int,
    ) -> None:
        ext = image_path.suffix
        success, encoded = cv2.imencode(ext, image)
        if not success:
            raise ValueError(f"{label} internal encode failed for: {image_path}")
        encoded.tofile(str(image_path))

        if not image_path.exists():
            raise IOError(f"Filesystem sync validation failed for {label}: {image_path}")

        nparr = np.fromfile(str(image_path), np.uint8)
        written = cv2.imdecode(nparr, read_flag)
        if written is None or written.size == 0:
            raise IOError(f"Read-back validation failed for {label}: {image_path}")

    def _bbox_iou(self, a: Dict[str, int], b: Dict[str, int]) -> float:
        ax1, ay1, aw, ah = a["x"], a["y"], a["w"], a["h"]
        bx1, by1, bw, bh = b["x"], b["y"], b["w"], b["h"]

        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter_area

        return float(inter_area / union) if union > 0 else 0.0

    def _get_masked_histogram(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask, [180, 128], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _should_reject_as_redundant(
        self,
        current_hist: np.ndarray,
        current_bbox: Optional[Dict[str, int]],
        last_hist: Optional[np.ndarray],
        last_bbox: Optional[Dict[str, int]],
    ) -> bool:
        if last_hist is None:
            return False

        hist_similarity = cv2.compareHist(last_hist, current_hist, cv2.HISTCMP_CORREL)

        bbox_iou = 0.0
        if current_bbox is not None and last_bbox is not None:
            bbox_iou = self._bbox_iou(last_bbox, current_bbox)

        return bool(
            hist_similarity > self.thresholds.min_similarity_score
            and bbox_iou > 0.85
        )

    def extract_keyframes(self, video_path: str, output_dir: str) -> tuple[List[str], Dict[str, Any]]:
        import os
        import hashlib
        import shutil
        from datetime import datetime
        from modules.operations.settings import settings, AppEnvironment

        # ── 1. Cache Key Calculation ──────────────────────────────────────────
        # Hash must include video content, parameters, and model versions
        video_p = Path(video_path)
        sha256 = "unknown"
        try:
            h = hashlib.sha256()
            with open(video_path, "rb") as f:
                # Read first 1MB and last 1MB for speed-efficient content hashing
                h.update(f.read(1024 * 1024))
                if video_p.stat().st_size > 2 * 1024 * 1024:
                    f.seek(-1024 * 1024, 2)
                    h.update(f.read(1024 * 1024))
            sha256 = h.hexdigest()
        except Exception:
            pass

        cache_data = {
            "sha256": sha256,
            "size": video_p.stat().st_size,
            "sample_rate": self.thresholds.frame_sample_rate,
            "min_blur": self.thresholds.min_blur_score,
            "min_similarity": self.thresholds.min_similarity_score,
            "seg_backend": self.seg_config.backend,
            "seg_model": getattr(self.seg_config, "rembg_model_name", "default"),
            "rembg_model": getattr(self.seg_config, "rembg_model_name", "u2net"),
            "rembg_threshold": getattr(self.seg_config, "rembg_mask_threshold", 50),
            "expected_product_color": getattr(settings, "expected_product_color", "unknown"),
            "segmentation_method": getattr(settings, "segmentation_method", "legacy"),
            "sam2_enabled": getattr(settings, "sam2_enabled", False),
        }
        current_hash = hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
        
        # ── 2. Skip Extraction (Cache) Logic ──────────────────────────────────
        skip_allowed = (os.getenv("SKIP_EXTRACTION") == "true" and settings.env == AppEnvironment.LOCAL_DEV)
        
        output_path = Path(output_dir)
        manifest_path = output_path / "extraction_manifest.json"

        if skip_allowed:
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r") as f:
                        cached_manifest = json.load(f)
                        if cached_manifest.get("manifest_hash") == current_hash:
                            existing_frames = sorted([str(f) for f in output_path.glob("*.jpg")])
                            if existing_frames:
                                logger.info(f"SKIP_EXTRACTION: Found {len(existing_frames)} valid frames. Skipping.")
                                return existing_frames, {"status": "skipped", "count": len(existing_frames), "verified": True}
                except Exception as e:
                    logger.warning(f"Cache manifest corrupted: {e}")
            logger.warning("SKIP_EXTRACTION requested but cache invalid/missing. Proceeding with extraction.")
        
        # ── 3. Stale Artifact Hygiene ─────────────────────────────────────────
        logger.info(f"Cleaning stale extraction artifacts in {output_path}...")
        
        # Files in root: *.jpg (frames)
        for f in output_path.glob("*.jpg"):
            f.unlink(missing_ok=True)
            
        # Masks and metadata
        masks_dir = output_path / "masks"
        if masks_dir.exists():
            for f in masks_dir.glob("*.png"): f.unlink(missing_ok=True)
            for f in masks_dir.glob("*.json"): f.unlink(missing_ok=True)
            
        # Masked previews
        masked_dir = output_path / "masked"
        if masked_dir.exists():
            for f in masked_dir.glob("*.jpg"): f.unlink(missing_ok=True)
            
        # Debug artifacts
        debug_dir = output_path / "debug"
        if debug_dir.exists():
            # Move to timestamped folder instead of simple delete
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_backup = output_path / f"debug_run_{ts}"
            try:
                shutil.move(str(debug_dir), str(debug_backup))
            except Exception:
                shutil.rmtree(debug_dir, ignore_errors=True)

        if manifest_path.exists():
            manifest_path.unlink()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        extracted_paths: List[str] = []
        frame_count = 0

        last_extracted_hist = None
        last_bbox = None

        rejection_counts = {
            "sampling": 0,
            "quality_or_mask": 0,
            "redundant_similarity": 0,
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        masks_dir = output_path / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Masked previews are useful for debugging/operator UX, but they should
        # not replace the raw frames used by COLMAP.
        masked_dir = output_path / "masked"
        masked_dir.mkdir(parents=True, exist_ok=True)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.thresholds.frame_sample_rate != 0:
                    rejection_counts["sampling"] += 1
                    frame_count += 1
                    continue

                mask, mask_meta = self.object_masker.generate_mask(frame)

                analysis = self.quality_analyzer.analyze_frame(frame, mask, mask_meta)
                if not analysis["overall_pass"]:
                    rejection_counts["quality_or_mask"] += 1
                    logger.debug(
                        f"Frame {frame_count} rejected :: reasons={analysis['failure_reasons']}"
                    )
                    frame_count += 1
                    continue

                current_hist = self._get_masked_histogram(frame, mask)
                current_bbox = mask_meta.get("bbox")

                if self._should_reject_as_redundant(
                    current_hist=current_hist,
                    current_bbox=current_bbox,
                    last_hist=last_extracted_hist,
                    last_bbox=last_bbox,
                ):
                    rejection_counts["redundant_similarity"] += 1
                    frame_count += 1
                    continue

                frame_base = f"frame_{len(extracted_paths):04d}"
                frame_filename = f"{frame_base}.jpg"
                frame_path = output_path / frame_filename
                mask_path = masks_dir / f"{frame_base}.png"
                meta_path = masks_dir / f"{frame_base}.json"

                focused_frame, focused_mask = self._prepare_object_centric_frame(
                    frame,
                    mask,
                    current_bbox,
                )

                # IMPORTANT FIX:
                # Reconstruction receives the raw frame, not the blacked-out preview.
                # This preserves background/table features for camera pose estimation.
                self._write_verified_image(frame_path, frame, "Raw extracted frame", cv2.IMREAD_COLOR)

                masked_frame_path = masked_dir / frame_filename
                self._write_verified_image(
                    masked_frame_path,
                    focused_frame,
                    "Masked preview frame",
                    cv2.IMREAD_COLOR,
                )

                self._write_verified_image(mask_path, focused_mask, "Mask", cv2.IMREAD_GRAYSCALE)

                with open(meta_path, "w", encoding="utf-8") as f:
                    clean_meta = {
                        k: v
                        for k, v in mask_meta.items()
                        if isinstance(v, (int, float, str, bool, type(None), dict, list))
                    }
                    json.dump(clean_meta, f, indent=2)

                if self.seg_config.debug_artifacts:
                    debug_dir = output_path / "debug"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imencode(".png", mask)[1].tofile(str(debug_dir / f"{frame_base}_raw_mask.png"))
                    cv2.imencode(".png", focused_mask)[1].tofile(str(debug_dir / f"{frame_base}_refined_mask.png"))

                extracted_paths.append(str(frame_path))
                last_extracted_hist = current_hist
                last_bbox = current_bbox

                if len(extracted_paths) >= self.config.max_frames:
                    break

                frame_count += 1
        finally:
            cap.release()

        # Dynamic color profile (replaces hardcoded EXPECTED_PRODUCT_COLOR=black)
        color_profile_dict: Dict[str, Any] = {}
        try:
            from modules.utils.color_profiler import resolve_color_profile

            profile = resolve_color_profile(
                expected_color_setting=getattr(settings, "expected_product_color", "auto"),
                frame_paths=[Path(p) for p in extracted_paths],
                masks_dir=masks_dir,
            )
            color_profile_dict = profile.to_dict()
            logger.info(
                f"[Extraction] color_profile: category={profile.category.value} "
                f"product_rgb={profile.product_rgb} bg_rgb={profile.background_rgb} "
                f"source={profile.source} confidence={profile.confidence:.2f}"
            )
        except Exception as e:
            logger.warning(f"Color profile detection failed, using fallback: {e}")

        # Capture profile (object size + scene type → pipeline thresholds)
        # Priority: session_capture_profile.json (UI-supplied) > env settings
        capture_profile_dict: Dict[str, Any] = {}
        try:
            from modules.operations.capture_profile import (
                resolve_from_setting, CaptureProfile,
            )

            session_profile_path = output_path.parent / "session_capture_profile.json"
            cp = None
            profile_source = "env"
            if session_profile_path.exists():
                try:
                    with open(session_profile_path, "r", encoding="utf-8") as pf:
                        cp = CaptureProfile.from_dict(json.load(pf))
                        profile_source = "session_manifest"
                except Exception as e:
                    logger.warning(f"session_capture_profile.json okunamadı, env fallback: {e}")

            if cp is None:
                cp = resolve_from_setting(
                    capture_profile_setting=getattr(settings, "capture_profile", "small_on_surface"),
                    material_hint=getattr(settings, "material_hint", "opaque"),
                )

            capture_profile_dict = cp.to_dict()
            capture_profile_dict["__source"] = profile_source
            logger.info(
                f"[Extraction] capture_profile: {cp.preset_key} "
                f"(source={profile_source}, material={cp.material_hint.value}, "
                f"poisson_depth={cp.recon_poisson_depth}, "
                f"mesh_budget={cp.recon_mesh_budget_faces}, "
                f"strip_planes={cp.remove_horizontal_planes})"
            )
        except Exception as e:
            logger.warning(f"Capture profile resolve failed, downstream uses defaults: {e}")

        extraction_report = {
            "total_frames_read": frame_count,
            "rejection_counts": rejection_counts,
            "saved_count": len(extracted_paths),
            "frame_mode": "raw_for_reconstruction",
            "masked_preview_dir": str(masked_dir),
            "masks_dir": str(masks_dir),
            "video_filename": Path(video_path).name,
            "timestamp": datetime.now().isoformat(),
            "color_profile": color_profile_dict,
            "capture_profile": capture_profile_dict,
        }

        logger.info(f"[Extraction] Product-aware summary for {extraction_report['video_filename']}:")
        logger.info(f"   - Total frames read: {frame_count}")
        logger.info(f"   - Rejected by quality/mask: {rejection_counts['quality_or_mask']}")
        logger.info(f"   - Rejected by similarity: {rejection_counts['redundant_similarity']}")
        logger.info(f"   - Total saved: {len(extracted_paths)}")
        logger.info("   - Frame mode: raw_for_reconstruction; masked previews saved separately")

        # Save manifest for future skips
        manifest_path = Path(output_dir) / "extraction_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump({
                "manifest_hash": current_hash,
                "video_path": video_path,
                "timestamp": str(datetime.now()),
                "frame_count": len(extracted_paths),
                "color_profile": color_profile_dict,
                "capture_profile": capture_profile_dict,
            }, f, indent=2)

        return extracted_paths, extraction_report
