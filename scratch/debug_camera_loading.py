import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from modules.asset_cleanup_pipeline.camera_projection import load_reconstruction_cameras, load_reconstruction_masks

def test_loading():
    job_id = "job_cap_29ab6fa1"
    # The actual path where the mesh is (from audit)
    # C:\Users\Lenovo\.gemini\antigravity\scratch\Qar_imageProccesing\data\reconstructions\job_cap_29ab6fa1\attempt_1_denser_frames\dense\meshed-poisson.ply
    
    mesh_path = Path(f"data/reconstructions/{job_id}/attempt_1_denser_frames/dense/meshed-poisson.ply")
    workspace_path = mesh_path.parent.parent
    
    print(f"Workspace Path: {workspace_path.absolute()}")
    print(f"Workspace exists: {workspace_path.exists()}")
    
    cameras = load_reconstruction_cameras(workspace_path)
    print(f"Loaded {len(cameras)} cameras")
    
    if cameras:
        names = [c["name"] for c in cameras[:5]]
        print(f"Sample camera names: {names}")
        
        masks = load_reconstruction_masks(workspace_path, [c["name"] for c in cameras])
        print(f"Loaded {len(masks)} masks")
    else:
        # Check if sparse folder exists
        sparse_dir = workspace_path / "sparse"
        print(f"Sparse dir exists: {sparse_dir.exists()}")
        if sparse_dir.exists():
            print(f"Contents of sparse: {[d.name for d in sparse_dir.iterdir()]}")
            for d in sparse_dir.iterdir():
                if d.is_dir():
                    print(f"Contents of sparse/{d.name}: {[f.name for f in d.iterdir()]}")

if __name__ == "__main__":
    test_loading()
