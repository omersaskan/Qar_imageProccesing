import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from modules.asset_cleanup_pipeline.cleaner import AssetCleaner

def finalize_manual_mesh():
    workspace = Path("workspace_quality_test")
    input_mesh = workspace / "meshed_manual_C.ply"
    
    if not input_mesh.exists():
        print(f"Error: {input_mesh} not found")
        return

    print(f"Finalizing mesh: {input_mesh}")
    
    # AssetCleaner expects a certain structure or can process a file
    # Based on worker.py, it takes data_root
    cleaner = AssetCleaner(data_root=str(workspace.absolute()))
    
    try:
        # We need to simulate the expected file locations if needed, 
        # or call a specific method of cleaner.
        # Let's check cleaner.py methods if possible, but usually process_cleanup() works.
        # But process_cleanup looks for dense/meshed-*.ply.
        
        # Let's copy our manual mesh to the expected location
        target = workspace / "dense/meshed-poisson.ply"
        import shutil
        shutil.copy2(input_mesh, target)
        print(f"Copied {input_mesh.name} to {target.name} for cleaner")
        
        # Also need a texture if available
        # If not, cleaner might use a default or fail.
        # Let's check if there is a texture
        texture = list(workspace.glob("*.png"))
        if texture:
            shutil.copy2(texture[0], workspace / "dense/texture.png")

        result = cleaner.process_cleanup(
            job_id="manual_relaxed_test",
            raw_mesh_path=str(target.absolute())
        )
        print(f"Cleanup result: {result}")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    finalize_manual_mesh()
