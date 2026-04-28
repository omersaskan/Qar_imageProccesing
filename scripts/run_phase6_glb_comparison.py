import os
import subprocess
import json
import sys
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
    print(f"<<< FINISHED EXPERIMENT: {name}\n")

def main():
    capture_id = "cap_29ab6fa1"
    
    # Experiment A: Legacy
    legacy_job_id = f"legacy_{capture_id}_compare"
    legacy_evidence_dir = ROOT / "evidence" / f"legacy_{capture_id}_glb_compare"
    legacy_env = {
        "SEGMENTATION_METHOD": "legacy",
        "SAM2_ENABLED": "false",
        "SAM2_MODE": "image"
    }
    
    # Experiment B: SAM2
    sam2_job_id = f"sam2_tiny_image_{capture_id}_compare"
    sam2_evidence_dir = ROOT / "evidence" / f"sam2_tiny_image_{capture_id}_glb_compare"
    sam2_env = {
        "SEGMENTATION_METHOD": "sam2",
        "SAM2_ENABLED": "true",
        "SAM2_MODEL_CFG": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "SAM2_CHECKPOINT": "models/sam2/sam2.1_hiera_tiny.pt",
        "SAM2_PROMPT_MODE": "manual_first_frame_box",
        "SAM2_REVIEW_ONLY": "true",
        "SAM2_MODE": "image"
    }

    try:
        # Run Legacy
        run_experiment("Legacy Baseline", legacy_env, capture_id, legacy_job_id, legacy_evidence_dir)
        
        # Run SAM2
        run_experiment("SAM2 Review-Only", sam2_env, capture_id, sam2_job_id, sam2_evidence_dir)
        
        print("\n" + "="*60)
        print("PHASE 6.1 GLB COMPARISON COMPLETED")
        print("="*60)
        print(f"Legacy Bundle: {legacy_evidence_dir}")
        print(f"SAM2 Bundle:   {sam2_evidence_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\n!!! EXPERIMENT FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
