import os
import subprocess
import json
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent.absolute()

def run_experiment(name, env_vars, capture_id, job_id, evidence_dir):
    print(f"\n>>> STARTING EXPERIMENT: {name}")
    env = os.environ.copy()
    env.update(env_vars)
    
    # Run Reconstruction
    cmd_recon = [
        "py", "run_real_recon.py",
        "--capture-id", capture_id,
        "--job-id", job_id
    ]
    print(f"Executing: {' '.join(cmd_recon)}")
    subprocess.run(cmd_recon, cwd=str(ROOT), env=env, check=True)
    
    # Run Evidence Export
    workspace = ROOT / "data" / "reconstructions" / job_id
    cmd_export = [
        "py", "scripts/export_reconstruction_evidence.py",
        "--job-id", job_id,
        "--workspace", str(workspace),
        "--output-dir", str(evidence_dir)
    ]
    print(f"Executing: {' '.join(cmd_export)}")
    subprocess.run(cmd_export, cwd=str(ROOT), env=env, check=True)
    
    # Copy texturing_metrics.json to evidence dir for easy access
    try:
        # It might be in recon/texturing or recon/attempt_0/texturing
        metrics_candidates = list(workspace.glob("**/texturing_metrics.json"))
        if metrics_candidates:
             shutil.copy2(metrics_candidates[0], evidence_dir / "texturing_metrics.json")
    except Exception as e:
        print(f"Warning: could not copy texturing_metrics.json: {e}")

    print(f"<<< FINISHED EXPERIMENT: {name}\n")

def main():
    capture_id = "cap_29ab6fa1"
    
    # Run A: Cream Neutralization (Current)
    job_a = f"cap_29ab6fa1_v5_cream"
    evidence_a = ROOT / "evidence_cap_29ab6fa1_v5" / "run_a_cream"
    env_a = {
        "SAM2_ENABLED": "false",
        "TEXTURE_NEUTRALIZATION_TYPE": "cream",
        "REQUIRE_TEXTURED_OUTPUT": "true",
        "EXPECTED_PRODUCT_COLOR": "white_cream"
    }
    
    # Run B: Black Mask Neutralization
    job_b = f"cap_29ab6fa1_v5_black"
    evidence_b = ROOT / "evidence_cap_29ab6fa1_v5" / "run_b_black_mask"
    env_b = {
        "SAM2_ENABLED": "false",
        "TEXTURE_NEUTRALIZATION_TYPE": "black_mask",
        "REQUIRE_TEXTURED_OUTPUT": "true",
        "EXPECTED_PRODUCT_COLOR": "white_cream"
    }

    try:
        # Run A
        run_experiment("A: Cream Neutralization", env_a, capture_id, job_a, evidence_a)
        
        # Run B
        run_experiment("B: Black Mask Neutralization", env_b, capture_id, job_b, evidence_b)
        
        print("\n" + "="*60)
        print("TEXTURE PATH AUDIT V5 COMPLETED")
        print("="*60)
        print(f"Run A (Cream):      {evidence_a}")
        print(f"Run B (Black Mask): {evidence_b}")
        print("="*60)
        
    except Exception as e:
        print(f"\n!!! AUDIT FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
