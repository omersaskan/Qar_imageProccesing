import numpy as np
from typing import List, Dict, Any, Optional

class GeometricAnalyzer:
    """
    Analyzes the geometric distribution and temporal continuity of captured frames.
    Focuses on detecting orbit gaps, viewpoint jumps, and parallax quality.
    """

    def __init__(self, 
                 gap_threshold: float = 0.25, 
                 continuity_threshold: float = 0.15,
                 min_viewpoint_spread: float = 0.6):
        self.gap_threshold = gap_threshold
        self.continuity_threshold = continuity_threshold
        self.min_viewpoint_spread = min_viewpoint_spread

    def analyze_orbit(self, signatures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes the trajectory and coverage based on frame signatures.
        """
        if not signatures:
            return {
                "span_x": 0.0,
                "max_gap": 0.0,
                "jump_ratio": 0.0,
                "hard_codes": [],
                "soft_codes": [],
                "all_codes": []
            }

        hard_codes = []
        soft_codes = []
        
        centers_x = [sig["center"][0] for sig in signatures]
        unique_centers_x = sorted(list(set(centers_x)))
        
        # 1. Horizontal Coverage & Gaps
        span_x = max(centers_x) - min(centers_x)
        
        # Detect large gaps in the center_x distribution
        gaps = []
        max_gap = 0.0
        for i in range(len(unique_centers_x) - 1):
            gap = unique_centers_x[i+1] - unique_centers_x[i]
            max_gap = max(max_gap, gap)
            if gap > self.gap_threshold:
                midpoint = (unique_centers_x[i+1] + unique_centers_x[i]) / 2
                if midpoint < 0.3:
                    gaps.append("LEFT_ORBIT_GAP")
                elif midpoint > 0.7:
                    gaps.append("RIGHT_ORBIT_GAP")
                else:
                    gaps.append("CENTRAL_ORBIT_GAP")

        if len(gaps) > 0:
            if len(gaps) > 2:
                hard_codes.append("LOW_HORIZONTAL_COVERAGE")
            else:
                if "LEFT_ORBIT_GAP" in gaps:
                    hard_codes.append("ORBIT_GAP_LEFT")
                if "RIGHT_ORBIT_GAP" in gaps:
                    hard_codes.append("ORBIT_GAP_RIGHT")
                if "CENTRAL_ORBIT_GAP" in gaps:
                    hard_codes.append("LOW_HORIZONTAL_COVERAGE")

        if span_x < self.min_viewpoint_spread:
             # User Request: SPREAD is always soft
             soft_codes.append("INSUFFICIENT_VIEWPOINT_SPREAD")

        # 2. Temporal Continuity (Sequence Jumps)
        jumps = 0
        for i in range(len(signatures) - 1):
            dist = np.linalg.norm(
                np.array(signatures[i+1]["center"]) - np.array(signatures[i]["center"])
            )
            if dist > self.continuity_threshold:
                jumps += 1
        
        jump_ratio = jumps / max(1, len(signatures) - 1)
        if jump_ratio > 0.2:
            # User Request: CONTINUITY is always soft
            soft_codes.append("WEAK_ORBIT_CONTINUITY")

        all_codes = sorted(list(set(hard_codes + soft_codes)))

        return {
            "span_x": float(span_x),
            "max_gap": float(max_gap),
            "jump_ratio": float(jump_ratio),
            "hard_codes": hard_codes,
            "soft_codes": soft_codes,
            "all_codes": all_codes
        }
