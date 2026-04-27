
import pytest
import json
from pathlib import Path
from scripts.evaluate_baselines import evaluate_pipeline_baselines

def test_evaluate_baselines_with_mock_data(tmp_path):
    job_dir = tmp_path / "job_test"
    job_dir.mkdir()
    
    evidence_data = {
        "delivery_checklist": {
            "final_status": "production_ready",
            "dense_image_count": 10,
            "dense_mask_exact_matches": 10,
            "dense_mask_dimension_matches": 10,
            "dense_mask_fallback_white_ratio": 0.0
        },
        "reports": {
            "cleanup_stats": {
                "isolation": {
                    "object_isolation_method": "mask_guided",
                    "isolation_confidence": 0.95
                }
            }
        }
    }
    
    with open(job_dir / "evidence_report.json", "w") as f:
        json.dump(evidence_data, f)
        
    results = evaluate_pipeline_baselines(tmp_path)
    
    assert results["total_jobs"] == 1
    assert results["status_distribution"]["production_ready"] == 1
    assert results["success_rate"] == 1.0
    assert results["mask_integrity"]["avg_exact_match_ratio"] == 1.0
    assert results["avg_isolation_confidence"] == 0.95

def test_evaluate_baselines_empty_root(tmp_path):
    results = evaluate_pipeline_baselines(tmp_path)
    assert results["status"] == "error"
    assert "No evidence reports found" in results["message"]
