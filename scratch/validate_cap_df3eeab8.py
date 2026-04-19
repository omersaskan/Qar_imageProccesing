import os
import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path("c:/modelPlate")
sys.path.append(str(project_root))

from modules.reconstruction_engine.runner import ReconstructionRunner
from modules.reconstruction_engine.job_manager import JobManager
from modules.shared_contracts.models import ReconstructionJobDraft

def run_validation():
    session_id = "cap_df3eeab8"
    capture_dir = project_root / "data/captures" / session_id
    frames_dir = capture_dir / "frames"
    
    if not frames_dir.exists():
        print(f"Error: Frames dir {frames_dir} not found")
        return

    input_frames = sorted([str(p) for p in frames_dir.glob("*.jpg")])
    print(f"Found {len(input_frames)} frames")

    # Initialize JobManager
    data_root = str(project_root / "data")
    manager = JobManager(data_root=data_root)

    # Create Job Draft
    draft = ReconstructionJobDraft(
        job_id=f"VAL_{session_id}",
        capture_session_id=session_id,
        input_frames=input_frames,
        product_id="PROD_VALIDATION",
    )

    try:
        job = manager.create_job(draft)
        print(f"Created job at {job.job_dir}")

        runner = ReconstructionRunner()
        # Ensure we use colmap_dense for this validation to test the new masking
        from modules.operations.settings import settings
        settings.recon_pipeline = "colmap_dense"
        settings.recon_fallback_steps = ["default"]
        
        print("Starting reconstruction...")
        manifest = runner.run(job)
        
        print("\n--- Reconstruction Successful ---")
        print(f"Job ID: {manifest.job_id}")
        print(f"Engine Type: {manifest.engine_type}")
        print(f"Mesh Path: {manifest.mesh_path}")
        print(f"Vertex Count: {manifest.mesh_metadata.vertex_count}")
        print(f"Face Count: {manifest.mesh_metadata.face_count}")
        
        # Pull additional stats from the audit
        audit_path = Path(job.job_dir) / "reconstruction_audit.json"
        if audit_path.exists():
            import json
            with open(audit_path, "r") as f:
                audit_data = json.load(f)
                best_attempt = audit_data["attempts"][audit_data["selected_best_index"]]
                print(f"Registered Images: {best_attempt.get('registered_images')}")
                print(f"Sparse Points: {best_attempt.get('sparse_points')}")
                print(f"Dense Points Fused: {best_attempt.get('dense_points_fused')}")
                print(f"Mesher Used: {best_attempt.get('mesher_used')}")

        # Check GLB export
        from modules.export_pipeline.glb_exporter import GLBExporter
        exporter = GLBExporter()
        glb_path = Path(job.job_dir) / "final_asset.glb"
        
        print(f"\nExporting GLB to {glb_path}...")
        export_res = exporter.export(
            mesh_path=manifest.mesh_path,
            output_path=str(glb_path),
            texture_path=manifest.texture_path,
        )
        print(f"GLB Export Result: {export_res['status']}")
        print(f"GLB Size: {export_res.get('filesize', 0)} bytes")

    except Exception as e:
        print(f"\nReconstruction Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_validation()
