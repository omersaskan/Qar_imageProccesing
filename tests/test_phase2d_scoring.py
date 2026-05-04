import pytest
from pathlib import Path
from modules.ai_3d_generation.candidate_selector import score_candidate, select_best

def test_score_background_removal_bonus():
    # Base candidate
    meta_base = {
        "provider_status": "ok",
        "output_glb_path": "dummy.glb",
        "preprocessing": {
            "enabled": True,
            "background_removed": False,
            "mask_source": "none",
            "bbox": [0, 0, 100, 100],
            "crop_bbox": [0, 0, 100, 100],
            "crop_width": 100,
            "crop_height": 100,
        },
        "warnings": []
    }
    # Create dummy GLB for scoring to pass existence check
    Path("dummy.glb").touch()
    
    score_base, _ = score_candidate(meta_base)
    
    # rembg success
    meta_rembg = {
        "provider_status": "ok",
        "output_glb_path": "dummy.glb",
        "preprocessing": {
            "enabled": True,
            "background_removed": True,
            "mask_source": "rembg",
            "bbox_source": "rembg_alpha",
            "bbox": [10, 10, 90, 90],
            "crop_bbox": [5, 5, 95, 95],
            "crop_width": 90,
            "crop_height": 90,
        },
        "warnings": []
    }
    score_rembg, breakdown = score_candidate(meta_rembg)
    
    assert score_rembg > score_base
    assert breakdown["background_removed_bonus"] == 8.0
    assert breakdown["rembg_bonus"] == 9.0  # 5 (mask_source) + 4 (bbox_source)

    Path("dummy.glb").unlink()

def test_score_foreground_ratio():
    def get_meta(ratio):
        return {
            "provider_status": "ok",
            "preprocessing": {
                "foreground_ratio_estimate": ratio,
                "bbox": [0, 0, 100, 100],
                "crop_bbox": [0, 0, 100, 100],
                "crop_width": 100,
                "crop_height": 100,
            }
        }
    
    # Ideal
    _, b_ideal = score_candidate(get_meta(0.30))
    assert b_ideal["foreground_ratio_score"] == 6.0
    
    # Good
    _, b_good = score_candidate(get_meta(0.10))
    assert b_good["foreground_ratio_score"] == 3.0
    
    # Too small
    _, b_small = score_candidate(get_meta(0.01))
    assert b_small["foreground_ratio_score"] == -8.0
    
    # Too large
    _, b_large = score_candidate(get_meta(0.95))
    assert b_large["foreground_ratio_score"] == -6.0

def test_score_fallback_penalties():
    def get_meta(crop_method, warning=None):
        return {
            "provider_status": "ok",
            "preprocessing": {
                "crop_method": crop_method,
                "bbox": [0, 0, 100, 100],
                "crop_bbox": [0, 0, 100, 100],
                "crop_width": 100,
                "crop_height": 100,
            },
            "warnings": [warning] if warning else []
        }
    
    # Fallback crop
    _, b_fallback = score_candidate(get_meta("fallback_center_crop"))
    assert b_fallback["fallback_penalty"] == -5.0
    
    # rembg failed
    _, b_failed = score_candidate(get_meta("fallback_center_crop", "rembg_failed_fallback_center_crop"))
    assert b_failed["fallback_penalty"] == -11.0 # -5 (crop_method) - 6 (warning)

    # rembg empty alpha
    _, b_empty = score_candidate(get_meta("fallback_center_crop", "rembg_empty_alpha_fallback_center_crop"))
    assert b_empty["fallback_penalty"] == -13.0 # -5 - 8

def test_bbox_sanity():
    def get_meta(bbox, crop_bbox, cw=100, ch=100, orig_w=500, orig_h=500):
        return {
            "provider_status": "ok",
            "preprocessing": {
                "bbox": bbox,
                "crop_bbox": crop_bbox,
                "crop_width": cw,
                "crop_height": ch,
                "original_width": orig_w,
                "original_height": orig_h,
            }
        }
    
    # Valid
    _, b_valid = score_candidate(get_meta([0,0,10,10], [0,0,12,12]))
    assert b_valid["bbox_sanity_score"] == 3.0
    
    # Missing bbox
    _, b_missing = score_candidate(get_meta(None, [0,0,12,12]))
    assert b_missing["bbox_sanity_score"] == -5.0
    
    # Zero crop
    _, b_zero = score_candidate(get_meta([0,0,10,10], [0,0,12,12], cw=0))
    # +3.0 (valid lists) - 5.0 (cw <= 0) = -2.0
    assert b_zero["bbox_sanity_score"] == -2.0
    
    # Tiny crop
    _, b_tiny = score_candidate(get_meta([0,0,10,10], [0,0,12,12], cw=5, ch=5, orig_w=1000, orig_h=1000))
    # 25 < 1000000 * 0.02 (20000)
    assert b_tiny["bbox_sanity_score"] == -1.0 # +3 (valid) - 4 (tiny)

def test_select_best_with_preprocessing_metrics():
    Path("cand1.glb").touch()
    Path("cand2.glb").touch()
    
    c1 = {
        "candidate_id": "cand_001",
        "provider_status": "ok",
        "status": "ok",
        "output_glb_path": "cand1.glb",
        "preprocessing": {
            "background_removed": True,
            "mask_source": "rembg",
            "foreground_ratio_estimate": 0.3
        }
    }
    c2 = {
        "candidate_id": "cand_002",
        "provider_status": "ok",
        "status": "ok",
        "output_glb_path": "cand2.glb",
        "preprocessing": {
            "background_removed": False,
            "crop_method": "fallback_center_crop"
        }
    }
    
    # Score them first
    s1, b1 = score_candidate(c1)
    s2, b2 = score_candidate(c2)
    c1["score"] = s1
    c1["score_breakdown"] = b1
    c2["score"] = s2
    c2["score_breakdown"] = b2
    
    best, ranking, reason = select_best([c1, c2])
    
    assert best["candidate_id"] == "cand_001"
    assert ranking[0]["candidate_id"] == "cand_001"
    assert ranking[0]["background_removed"] is True
    assert ranking[1]["background_removed"] is False
    assert ranking[0]["mask_source"] == "rembg"
    assert ranking[0]["foreground_ratio_estimate"] == 0.3
    
    Path("cand1.glb").unlink()
    Path("cand2.glb").unlink()
