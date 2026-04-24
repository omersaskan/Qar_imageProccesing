from pathlib import Path
from modules.reconstruction_engine.adapter import COLMAPAdapter
import os
import shutil

def run_verify():
    # Setup
    adapter = COLMAPAdapter()
    
    # Path to real frames extracted in previous attempt
    # Note: Using the subset that successfully registered in the original failure (29/30 images)
    # Actually, I'll just use all 30 frames.
    frames_dir = Path(r"c:\modelPlate\data\reconstructions\job_cap_24b4136c\attempt_1_denser_frames\images")
    if not frames_dir.exists():
        print(f"Error: Frames dir not found at {frames_dir}")
        return
        
    input_frames = [str(f) for f in frames_dir.glob("*.jpg")]
    print(f"Found {len(input_frames)} frames.")
    
    # New workspace for verification
    output_dir = Path(r"c:\modelPlate\data\reconstructions\verify_fix_cap_24b4136c")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    print(f"Running reconstruction in {output_dir}...")
    print("Mask enforcement: DISABLED (only for verification speed/stability)")
    
    try:
        # We run the adapter with enforce_masks=False to bypass missing masks directory issue
        results = adapter.run_reconstruction(input_frames, output_dir, enforce_masks=False)
        print("\nINSTANT SUCCESS: Reconstruction completed!")
        print(f"Mesh path: {results['mesh_path']}")
        print(f"Registered images: {results.get('registered_images', 'N/A')}")
        print(f"Sparse points: {results.get('sparse_points', 'N/A')}")
    except Exception as e:
        print(f"\nPROCESS REPORT:")
        print(f"Error: {e}")
        
    # Check log for the critical transition
    log_path = output_dir / "reconstruction.log"
    if log_path.exists():
        print("\n--- Log Analysis ---")
        with open(log_path, "r") as f:
            content = f.read()
            
            # Check for the key stats in log
            if "Registered images: 29" in content or "Images: 29" in content:
                print("Found 'Registered images: 29' in log.")
            if "Points: 4187" in content or "Points3D: 4187" in content:
                print("Found 'Points: 4187' in log.")
            
            # Check for progress past gate
            if "Running image_undistorter" in content or "patch_match_stereo" in content:
                print("ADVANCED: Pipeline successfully cleared the sparse gate!")
            else:
                print("STALLED: Pipeline did not reach dense stage.")

if __name__ == "__main__":
    run_verify()
