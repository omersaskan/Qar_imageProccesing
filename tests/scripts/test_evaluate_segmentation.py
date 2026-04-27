
import pytest
import numpy as np
import cv2
import json
from pathlib import Path
from scripts.evaluate_segmentation import evaluate_masks

def test_evaluate_masks_perfect_match(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    
    # Create perfect match mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    
    cv2.imwrite(str(gt_dir / "test.png"), mask)
    cv2.imwrite(str(pred_dir / "test.png"), mask)
    
    results = evaluate_masks(gt_dir, pred_dir)
    
    assert results["status"] == "success"
    assert results["avg_iou"] == 1.0
    assert results["avg_leakage_ratio"] == 0.0

def test_evaluate_masks_partial_overlap(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    
    # GT: 50x50 square at (0,0)
    gt_mask = np.zeros((100, 100), dtype=np.uint8)
    gt_mask[0:50, 0:50] = 255
    
    # Pred: 50x50 square at (25,0) -> 25x50 overlap
    pred_mask = np.zeros((100, 100), dtype=np.uint8)
    pred_mask[0:50, 25:75] = 255
    
    cv2.imwrite(str(gt_dir / "test.png"), gt_mask)
    cv2.imwrite(str(pred_dir / "test.png"), pred_mask)
    
    results = evaluate_masks(gt_dir, pred_dir)
    
    # Intersection = 25 * 50 = 1250
    # Union = (50 * 50) + (50 * 50) - 1250 = 3750
    # IoU = 1250 / 3750 = 0.333...
    # Leakage = (Pixels in Pred but not GT) / GT area = (25 * 50) / (50 * 50) = 0.5
    
    assert results["status"] == "success"
    assert pytest.approx(results["avg_iou"], 0.01) == 0.333
    assert pytest.approx(results["avg_leakage_ratio"], 0.01) == 0.5
