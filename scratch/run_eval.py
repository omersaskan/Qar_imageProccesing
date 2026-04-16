import sys
import os
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sys.path.append(str(Path(__file__).parent.parent))

from modules.shared_contracts.models import ReconstructionJob
from modules.reconstruction_engine.runner import ReconstructionRunner
from modules.operations.settings import settings

def main():
    job_dir = Path(r"c:\modelPlate\data\reconstructions\job_cap_9b6ff69b")
    job_json = job_dir / "job.json"
    
    with open(job_json, "r") as f:
        job_data = json.load(f)
    
    # We use eval4 to bypass the pollution from previous iterations
    job_dir_eval = Path(r"c:\modelPlate\data\reconstructions\job_cap_9b6ff69b_eval4")
    if job_dir_eval.exists():
        shutil.rmtree(job_dir_eval)
        
    job_data["job_id"] = "job_cap_9b6ff69b_eval4"
    job_data["job_dir"] = str(job_dir_eval)
    
    settings.recon_fallback_sample_rate = 5 
    settings.recon_fallback_steps = ["denser_frames"]
    
    job = ReconstructionJob(**job_data)
    runner = ReconstructionRunner()
    
    try:
        manifest = runner.run(job)
        print("\n\nTEST PASSED")
        print(f"Manifest created at: {manifest.mesh_path}")
        print(f"Vertices: {manifest.mesh_metadata.vertex_count}")
        print(f"Faces: {manifest.mesh_metadata.face_count}")
    except Exception as e:
        print("\n\nTEST FAILED")
        print(str(e))

if __name__ == "__main__":
    main()
