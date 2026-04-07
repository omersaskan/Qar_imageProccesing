from typing import List, Dict, Any
import numpy as np

class CoverageAnalyzer:
    def __init__(self):
        pass

    def analyze_coverage(self, extracted_frames: List[str]) -> Dict[str, Any]:
        """
        Perform heuristic-based coverage estimation.
        """
        num_frames = len(extracted_frames)
        
        # 1. Diversity check (heuristic: if num_frames is low, diversity is likely low)
        diversity_status = "sufficient" if num_frames >= 20 else "insufficient"
        
        # 2. Top-down missing (heuristic: just a placeholder logic for now)
        # In a real scenario, this would use a viewpoint detection model.
        top_down_missing = num_frames < 30 # Just a placeholder
        
        reasons = []
        if num_frames < 10:
            reasons.append("Too few high-quality frames extracted.")
        if top_down_missing:
            reasons.append("Top-down angle might be missing.")
        if num_frames < 30:
            reasons.append("Insufficient side view diversity suspected.")

        return {
            "num_frames": num_frames,
            "diversity": diversity_status,
            "top_down_captured": not top_down_missing,
            "overall_status": "sufficient" if num_frames >= 30 else "insufficient",
            "reasons": reasons
        }
