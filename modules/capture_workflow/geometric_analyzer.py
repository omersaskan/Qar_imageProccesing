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
            return {"status": "unknown", "codes": []}

        codes = []
        centers_x = [sig["center"][0] for sig in signatures]
        unique_centers_x = sorted(list(set(centers_x)))
        
        # 1. Horizontal Coverage & Gaps
        span_x = max(centers_x) - min(centers_x)
        
        # Detect large gaps in the center_x distribution
        gaps = []
        for i in range(len(unique_centers_x) - 1):
            gap = unique_centers_x[i+1] - unique_centers_x[i]
            if gap > self.gap_threshold:
                # If gap is near the edges, we can be more specific
                midpoint = (unique_centers_x[i+1] + unique_centers_x[i]) / 2
                if midpoint < 0.3:
                    gaps.append("LEFT_ORBIT_GAP")
                elif midpoint > 0.7:
                    gaps.append("RIGHT_ORBIT_GAP")
                else:
                    gaps.append("CENTRAL_ORBIT_GAP")

        if len(gaps) > 0:
            # Be cautious as requested: use generic code if too many gaps
            if len(gaps) > 2:
                codes.append("LOW_HORIZONTAL_COVERAGE")
            else:
                # Map specific gaps to requested codes (ORBIT_GAP_LEFT/RIGHT)
                if "LEFT_ORBIT_GAP" in gaps:
                    codes.append("ORBIT_GAP_LEFT")
                if "RIGHT_ORBIT_GAP" in gaps:
                    codes.append("ORBIT_GAP_RIGHT")
                if "CENTRAL_ORBIT_GAP" in gaps:
                    codes.append("LOW_HORIZONTAL_COVERAGE")

        if span_x < self.min_viewpoint_spread:
             codes.append("INSUFFICIENT_VIEWPOINT_SPREAD")

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
            codes.append("WEAK_ORBIT_CONTINUITY")

        return {
            "span_x": float(span_x),
            "max_gap": float(max([unique_centers_x[i+1] - unique_centers_x[i] for i in range(len(unique_centers_x)-1)]) if len(unique_centers_x) > 1 else 0),
            "jump_ratio": float(jump_ratio),
            "codes": list(set(codes))
        }
