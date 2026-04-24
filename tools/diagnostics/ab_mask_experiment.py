import argparse
import sys
import os
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any

from modules.operations.settings import settings
from modules.reconstruction_engine.adapter import COLMAPAdapter, ColmapCommandBuilder

class UnmaskedCommandBuilder(ColmapCommandBuilder):
    """Overrides feature_extractor to OMIT masks."""
    def feature_extractor(self, db_path: Path, images_dir: Path, masks_dir: Path, max_size: int) -> List[str]:
        cmd = [
            self.bin, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            # NO mask_path here
            "--FeatureExtraction.use_gpu", "1" if self.use_gpu else "0",
            "--FeatureExtraction.max_image_size", str(max_size),
        ]
        return cmd

class ExperimentRunner:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.data_root = Path(settings.data_root).resolve()
        self.captures_dir = self.data_root / "captures"
        self.session_dir = self.captures_dir / session_id
        self.frames_dir = self.session_dir / "frames"
        
        # Experiment workspaces
        self.exp_root = self.data_root / "experiments" / session_id
        self.exp_root.mkdir(parents=True, exist_ok=True)
        
    def get_input_frames(self) -> List[str]:
        frames = list(self.frames_dir.glob("*.jpg"))
        return [str(f) for f in sorted(frames)]

    def run_masked(self) -> Dict[str, Any]:
        print(f"\n>>> Running EXPERIMENT A: MASKED (Session: {self.session_id})")
        output_dir = self.exp_root / "run_masked"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        
        adapter = COLMAPAdapter() # Uses standard standard builder
        try:
            result = adapter.run_reconstruction(self.get_input_frames(), output_dir)
            result["status"] = "SUCCESS"
            return result
        except Exception as e:
            print(f"Masked run failed: {e}")
            return {"status": "FAILED", "error": str(e)}

    def run_unmasked(self) -> Dict[str, Any]:
        print(f"\n>>> Running EXPERIMENT B: UNMASKED (Session: {self.session_id})")
        output_dir = self.exp_root / "run_unmasked"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        
        adapter = COLMAPAdapter()
        # Inject unmasked builder
        adapter.builder = UnmaskedCommandBuilder(adapter._engine_path, adapter._use_gpu)
        
        try:
            result = adapter.run_reconstruction(self.get_input_frames(), output_dir)
            result["status"] = "SUCCESS"
            return result
        except Exception as e:
            print(f"Unmasked run failed: {e}")
            return {"status": "FAILED", "error": str(e)}

    def report(self, res_a: Dict[str, Any], res_b: Dict[str, Any]):
        print("\n\n" + "="*50)
        print("DIAGNOSTIC A/B EXPERIMENT REPORT")
        print("="*50)
        
        headers = ["Metric", "RUN A (Masked)", "RUN B (Unmasked)"]
        
        def fmt(val):
            return str(val) if val is not None else "N/A"

        metrics = [
            ("Status", "status"),
            ("Accepted Frames", "registered_images"), # Approximation for prep
            ("Registered Images", "registered_images"),
            ("Sparse Points", "sparse_points"),
            ("Selected Model", "selected_sparse_model"),
            ("Dense Points (Fused)", "dense_points_fused"),
            ("Mesher Used", "mesher_used"),
            ("Workspace", "log_path"),
        ]
        
        print(f"| {headers[0]:<20} | {headers[1]:<20} | {headers[2]:<20} |")
        print(f"| {'-'*20} | {'-'*20} | {'-'*20} |")
        
        for label, key in metrics:
            val_a = fmt(res_a.get(key))
            val_b = fmt(res_b.get(key))
            # truncate workspace path for readability
            if key == "log_path":
                val_a = Path(val_a).parent.name
                val_b = Path(val_b).parent.name
            print(f"| {label:<20} | {val_a:<20} | {val_b:<20} |")

        # Conclusion Logic
        reg_a = res_a.get("registered_images", 0) or 0
        reg_b = res_b.get("registered_images", 0) or 0
        
        print("\nConclusion:")
        if res_b["status"] == "FAILED" and res_a["status"] == "FAILED":
            print(">>> CAPTURE BOTTLENECK: Both runs failed to register sufficient images.")
        elif reg_b > reg_a * 2 and reg_b > 10:
            print(f">>> MASKING BOTTLENECK: Unmasked run registered significantly more images ({reg_b} vs {reg_a}).")
        elif reg_b <= reg_a + 2 and reg_b < 10:
            print(">>> CAPTURE BOTTLENECK: Removing masks did not improve registration significantly.")
        else:
            print(">>> MIXED BOTTLENECK: Both masking and capture quality are likely limiting factors.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.session_id)
    res_a = runner.run_masked()
    res_b = runner.run_unmasked()
    
    runner.report(res_a, res_b)

if __name__ == "__main__":
    main()
