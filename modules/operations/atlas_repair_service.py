import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("atlas_repair")

class AtlasRepairService:
    def __init__(self, analyzer=None):
        from modules.qa_validation.texture_quality import TextureQualityAnalyzer
        self.analyzer = analyzer or TextureQualityAnalyzer()

    def repair_atlas(self, atlas_path: str, expected_color: str = "unknown") -> Dict[str, Any]:
        """
        Attempts to improve atlas quality via post-processing if it failed due to darkness.
        """
        p = Path(atlas_path)
        if not p.exists():
            return {"status": "error", "reason": "file_not_found"}

        # 1. Initial Analysis
        initial_stats = self.analyzer.analyze_path(atlas_path, expected_product_color=expected_color)
        
        if initial_stats["texture_quality_status"] == "success":
            return {"status": "no_repair_needed", "stats": initial_stats}

        # Check if the failure is mainly due to darkness
        reasons = initial_stats.get("texture_quality_reasons", [])
        is_dark = any("black" in r.lower() or "luminance" in r.lower() or "color match failed" in r.lower() for r in reasons)
        is_contaminated = any("background" in r.lower() or "neon" in r.lower() or "flat" in r.lower() for r in reasons)

        if not is_dark or is_contaminated:
            logger.info("Atlas repair skipped: not a darkness issue or too contaminated.")
            return {"status": "repair_skipped", "stats": initial_stats}

        # 2. Attempt Correction (Exposure + Gamma)
        img = cv2.imread(atlas_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"status": "error", "reason": "load_failed"}

        # Convert to float for processing
        # Separate alpha if exists
        has_alpha = img.shape[2] == 4 if len(img.shape) == 3 else False
        if has_alpha:
            alpha = img[:, :, 3]
            rgb = img[:, :, :3].astype(np.float32)
        else:
            rgb = img.astype(np.float32)

        # A) Auto Exposure (Stretch to 95th percentile)
        p95 = np.percentile(rgb, 95)
        if p95 > 10: # Only if there is SOME signal
            scale = 230.0 / p95
            rgb_scaled = np.clip(rgb * scale, 0, 255)
        else:
            rgb_scaled = rgb

        # B) Gamma Correction (Lighter)
        gamma = 1.2
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        rgb_gamma = cv2.LUT(rgb_scaled.astype(np.uint8), table)

        # Reassemble
        if has_alpha:
            corrected_img = cv2.merge([rgb_gamma[:,:,0], rgb_gamma[:,:,1], rgb_gamma[:,:,2], alpha])
        else:
            corrected_img = rgb_gamma

        # 3. Verify
        corrected_path = p.parent / (p.stem + "_corrected" + p.suffix)
        cv2.imwrite(str(corrected_path), corrected_img)
        
        corrected_stats = self.analyzer.analyze_path(str(corrected_path), expected_product_color=expected_color)
        
        # 4. Decision: Use only if it's better
        if self._is_better(corrected_stats, initial_stats):
            logger.info(f"Atlas repair successful! New grade: {corrected_stats.get('texture_quality_grade')}")
            # Overwrite original atlas with corrected version for downstream?
            # Or just return the new path. The user said "save texture_atlas_corrected.jpg".
            return {
                "status": "repaired",
                "original_stats": initial_stats,
                "repaired_stats": corrected_stats,
                "repaired_path": str(corrected_path)
            }
        else:
            logger.info("Atlas repair failed to improve quality enough.")
            if corrected_path.exists():
                corrected_path.unlink()
            return {"status": "repair_failed", "stats": initial_stats}

    def _is_better(self, new: Dict[str, Any], old: Dict[str, Any]) -> bool:
        # Better grade is a win
        grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        if grade_map.get(new.get("texture_quality_grade", "F"), 0) > grade_map.get(old.get("texture_quality_grade", "F"), 0):
            return True
        
        # Or same grade but improved match score significantly
        if new.get("expected_product_color_match_score", 0) > old.get("expected_product_color_match_score", 0) + 0.2:
            # But don't introduce neon artifacts
            if new.get("neon_artifact_ratio", 0) <= old.get("neon_artifact_ratio", 0) + 0.01:
                 return True
                 
        return False
