import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import List

# Dynamic project root discovery
# Assumes script is in <root>/scratch/
# Dynamic project root discovery
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from modules.reconstruction_engine.runner import ReconstructionRunner
from modules.reconstruction_engine.job_manager import JobManager
from modules.shared_contracts.models import ReconstructionJobDraft
from modules.operations.settings import settings

def run_validation():
    parser = argparse.ArgumentParser(description="3D Reconstruction Validation Script")
    parser.add_argument("--session-id", type=str, default="cap_df3eeab8", help="Capture session ID")
    parser.add_argument("--pipeline", type=str, default=None, help="Reconstruction pipeline (overrides env)")
    parser.add_argument("--data-root", type=str, default=None, help="Data root path (overrides env)")
    parser.add_argument("--skip-glb", action="store_true", help="Skip GLB export stage")
    
    args = parser.parse_args()

    # Configuration Priority: CLI Arg > Environment Variable > Settings Default
    effective_data_root = args.data_root or os.getenv("DATA_ROOT") or str(project_root / "data")
    
    # Update settings singleton for this run
    settings.data_root = effective_data_root
    
    # SPRINT 3: Do NOT override recon_pipeline or fallback_steps here
    # to ensure we test the actual system/env configuration.
    effective_pipeline = settings.recon_pipeline
    
    session_id = args.session_id
    data_root_path = Path(effective_data_root)
    capture_dir = data_root_path / "captures" / session_id
    frames_dir = capture_dir / "frames"
    
    print("=" * 50)
    print("      RECONSTRUCTION VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Project Root:     {project_root}")
    print(f"Data Root:        {effective_data_root}")
    print(f"Session ID:       {session_id}")
    print(f"Pipeline:         {effective_pipeline}")
    print(f"Environment:      {settings.env.value}")
    print("-" * 50)

    if not frames_dir.exists():
        print(f"Error: Frames dir {frames_dir} not found")
        return

    input_frames = sorted([str(p) for p in frames_dir.glob("*.jpg")])
    if not input_frames:
        print(f"Error: No frames found in {frames_dir}")
        return
        
    print(f"Found {len(input_frames)} frames")

    # Initialize JobManager
    manager = JobManager(data_root=effective_data_root)

    # Create Unique Job ID for this validation run (timestamp suffix)
    timestamp = int(time.time())
    unique_job_id = f"VAL_{session_id}_{timestamp}"

    # Create Job Draft
    draft = ReconstructionJobDraft(
        job_id=unique_job_id,
        capture_session_id=session_id,
        input_frames=input_frames,
        product_id="PROD_VALIDATION",
    )

    try:
        job = manager.create_job(draft)
        print(f"Created job: {unique_job_id}")
        print(f"Directory:  {job.job_dir}")

        runner = ReconstructionRunner()
        
        print("\nStarting reconstruction process...")
        start_time = time.time()
        manifest = runner.run(job)
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("--- RECONSTRUCTION SUCCESSFUL ---")
        print(f"Job ID:          {manifest.job_id}")
        print(f"Engine Type:     {manifest.engine_type}")
        print(f"Time Taken:      {elapsed:.2f}s")
        print(f"Mesh Path:       {manifest.mesh_path}")
        print(f"Vertex Count:    {manifest.mesh_metadata.vertex_count}")
        print(f"Face Count:      {manifest.mesh_metadata.face_count}")
        
        # Pull additional stats from the audit
        audit_path = Path(job.job_dir) / "reconstruction_audit.json"
        if audit_path.exists():
            with open(audit_path, "r") as f:
                audit_data = json.load(f)
                best_attempt = audit_data["attempts"][audit_data["selected_best_index"]]
                print("-" * 50)
                print(f"Registered Images:   {best_attempt.get('registered_images')}")
                print(f"Sparse Points:       {best_attempt.get('sparse_points')}")
                print(f"Dense Points Fused:  {best_attempt.get('dense_points_fused')}")
                print(f"Mesher Used:         {best_attempt.get('mesher_used')}")
        print("=" * 50 + "\n")

        # Check GLB export
        if not args.skip_glb:
            from modules.export_pipeline.glb_exporter import GLBExporter
            exporter = GLBExporter()
            glb_path = Path(job.job_dir) / "final_asset.glb"
            
            print(f"Exporting GLB to {glb_path}...")
            export_res = exporter.export(
                mesh_path=manifest.mesh_path,
                output_path=str(glb_path),
                texture_path=manifest.texture_path,
            )
            print(f"GLB Export Result: {export_res['status']}")
            print(f"GLB Size: {export_res.get('filesize', 0)} bytes")

    except Exception as e:
        print(f"\n" + "!" * 50)
        print(f"RECONSTRUCTION FAILED: {e}")
        print("!" * 50)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_validation()
