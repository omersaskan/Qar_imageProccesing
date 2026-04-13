import argparse
import sys
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any

from modules.operations.settings import settings
from modules.reconstruction_engine.adapter import COLMAPAdapter

class DensityExperimentTool:
    """
    Diagnostic tool to compare reconstruction quality across different frame densities.
    Helps determine if 'too few frames' is the primary bottleneck.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.data_root = Path(settings.data_root).resolve()
        self.session_dir = self.data_root / "captures" / session_id
        self.frames_dir = self.session_dir / "frames"
        
        self.exp_root = self.data_root / "experiments" / "density" / session_id
        self.exp_root.mkdir(parents=True, exist_ok=True)

    def run_experiment(self, sampling_rates: List[int]):
        """
        Runs reconstruction for each sampling rate (e.g. 10 means 'every 10th frame').
        """
        all_frames = sorted(list(self.frames_dir.glob("*.jpg")))
        if not all_frames:
            print(f"Error: No frames found for session {self.session_id}")
            return

        results = []
        for rate in sampling_rates:
            print(f"\n>>> Testing Density: Every {rate}th frame")
            subset = all_frames[::rate]
            print(f"    Subsampled to {len(subset)} frames")
            
            output_dir = self.exp_root / f"rate_{rate}"
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
            
            adapter = COLMAPAdapter()
            try:
                res = adapter.run_reconstruction([str(f) for f in subset], output_dir)
                res["rate"] = rate
                res["total_input"] = len(subset)
                res["status"] = "SUCCESS"
                results.append(res)
            except Exception as e:
                print(f"    Failed for rate {rate}: {e}")
                results.append({"rate": rate, "total_input": len(subset), "status": "FAILED", "error": str(e)})

        self.print_report(results)

    def print_report(self, results: List[Dict[str, Any]]):
        print("\n\n" + "="*60)
        print("CAPTURE DENSITY DIAGNOSTIC REPORT")
        print("="*60)
        print(f"| Rate | Input | Reg. Img | Registration % | Points | Status |")
        print(f"|------|-------|----------|----------------|--------|--------|")
        
        for r in results:
            rate = r["rate"]
            inp = r["total_input"]
            reg = r.get("registered_images", 0)
            pts = r.get("sparse_points", 0)
            status = r["status"]
            ratio = (reg / inp * 100) if inp > 0 else 0
            
            print(f"| {rate:<4} | {inp:<5} | {reg:<8} | {ratio:>13.1f}% | {pts:<6} | {status:<6} |")

        print("\nConclusion:")
        if len(results) >= 2:
            r1 = results[0] # lowest rate (highest density)
            r2 = results[-1] # highest rate (lowest density)
            
            if r1.get("registered_images", 0) > r2.get("registered_images", 0) + 5:
                print(">>> HIGHER DENSITY HELPS: Registration improved significantly with more frames.")
                print("    Consider decreasing config.frame_sample_rate for this session type.")
            elif r1["status"] == "FAILED" and r2["status"] == "FAILED":
                print(">>> CRITICAL CAPTURE ERROR: Neither density resolved the registration failure.")
                print("    Likely cause: Poor overlap, blur, or erratic camera motion.")
            else:
                print(">>> DENSITY IS NOT THE BOTTLENECK: Registration ratios are comparable.")
                print("    Focus on capture geometry (orbit continuity) instead.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--rates", nargs="+", type=int, default=[5, 10, 20])
    args = parser.parse_args()
    
    tool = DensityExperimentTool(args.session_id)
    tool.run_experiment(args.rates)

if __name__ == "__main__":
    main()
