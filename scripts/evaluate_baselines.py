
import os
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

def evaluate_pipeline_baselines(evidence_root: Path) -> Dict[str, Any]:
    """
    Analyzes multiple evidence reports to establish baseline performance.
    """
    report_paths = list(evidence_root.glob("**/evidence_report.json"))
    
    total_jobs = len(report_paths)
    if total_jobs == 0:
        return {"status": "error", "message": f"No evidence reports found in {evidence_root}"}
        
    statuses = []
    isolation_methods = []
    isolation_confidences = []
    dense_mask_integrity_stats = {
        "exact_match_ratio": [],
        "dim_match_ratio": [],
        "fallback_white_ratio": []
    }
    
    for path in report_paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                
            checklist = data.get("delivery_checklist", {})
            statuses.append(checklist.get("final_status", "unknown"))
            
            reports = data.get("reports", {})
            cleanup = reports.get("cleanup_stats", {})
            isolation = cleanup.get("isolation", {})
            
            isolation_methods.append(isolation.get("object_isolation_method", "unknown"))
            isolation_confidences.append(isolation.get("isolation_confidence", 0.0))
            
            # Mask integrity
            total_dense = checklist.get("dense_image_count", 0)
            if total_dense > 0:
                exact = checklist.get("dense_mask_exact_matches", 0)
                dim = checklist.get("dense_mask_dimension_matches", 0)
                fallback = checklist.get("dense_mask_fallback_white_ratio", 1.0)
                
                dense_mask_integrity_stats["exact_match_ratio"].append(exact / total_dense)
                dense_mask_integrity_stats["dim_match_ratio"].append(dim / total_dense)
                dense_mask_integrity_stats["fallback_white_ratio"].append(fallback)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    status_dist = Counter(statuses)
    method_dist = Counter(isolation_methods)
    
    return {
        "total_jobs": total_jobs,
        "status_distribution": dict(status_dist),
        "isolation_method_distribution": dict(method_dist),
        "avg_isolation_confidence": float(sum(isolation_confidences) / len(isolation_confidences)) if isolation_confidences else 0.0,
        "mask_integrity": {
            "avg_exact_match_ratio": float(sum(dense_mask_integrity_stats["exact_match_ratio"]) / len(dense_mask_integrity_stats["exact_match_ratio"])) if dense_mask_integrity_stats["exact_match_ratio"] else 0.0,
            "avg_dim_match_ratio": float(sum(dense_mask_integrity_stats["dim_match_ratio"]) / len(dense_mask_integrity_stats["dim_match_ratio"])) if dense_mask_integrity_stats["dim_match_ratio"] else 0.0,
            "avg_fallback_white_ratio": float(sum(dense_mask_integrity_stats["fallback_white_ratio"]) / len(dense_mask_integrity_stats["fallback_white_ratio"])) if dense_mask_integrity_stats["fallback_white_ratio"] else 0.0,
        },
        "success_rate": float((status_dist["production_ready"] + status_dist["review_ready"]) / total_jobs) if total_jobs > 0 else 0.0
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Pipeline Baselines")
    parser.add_argument("--root", required=True, help="Root directory containing evidence bundles")
    args = parser.parse_args()
    
    baselines = evaluate_pipeline_baselines(Path(args.root))
    print(json.dumps(baselines, indent=2))
