import os
import sys
import json
import traceback
from pathlib import Path
from modules.reconstruction_engine.adapter import COLMAPAdapter
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType
from modules.export_pipeline.glb_exporter import GLBExporter

def run_test(name, num_images, max_image_size):
    print(f"\n{'='*60}")
    print(f"RUNNING TEST: {name}")
    print(f"Images: {num_images}, Max Size: {max_image_size}")
    print(f"{'='*60}")
    
    workspace = Path(f"workspace_{name.lower().replace(' ', '_')}")
    workspace = workspace.absolute()
    if workspace.exists():
        import shutil
        shutil.rmtree(workspace)
    workspace.mkdir(exist_ok=True)
    
    images_source = Path("scratch/test_extraction_output").absolute()
    all_images = sorted(list(images_source.glob("*.jpg")))
    input_frames = [str(p) for p in all_images[:num_images]]
    
    if not input_frames:
        print("No images found!")
        return False

    # Set environment for max image size
    os.environ["RECON_MAX_IMAGE_SIZE"] = str(max_image_size)
    os.environ["RECON_GPU_INDEX"] = "0"
    
    try:
        # Step 1: Reconstruction
        adapter = COLMAPAdapter()
        results = adapter.run_reconstruction(input_frames, workspace)
        print("Reconstruction finished.")
        
        # Step 2: Cleanup
        print("Starting cleanup...")
        cleaner = AssetCleaner(data_root=str(workspace))
        metadata, cleanup_stats, cleaned_mesh_path = cleaner.process_cleanup(
            job_id=name,
            raw_mesh_path=results["mesh_path"],
            profile_type=CleanupProfileType.MOBILE_DEFAULT
        )
        print("Cleanup finished.")
        
        # Step 3: GLB Export
        print("Starting GLB export...")
        exporter = GLBExporter()
        glb_path = workspace / "output.glb"
        export_result = exporter.export(
            mesh_path=cleaned_mesh_path,
            output_path=str(glb_path),
            profile_name="standard"
        )
        print("GLB Export finished.")
        
        # Report Metrics
        print(f"\n--- METRICS FOR {name} ---")
        
        # 1. Depth maps count
        dm_dir = workspace / "dense" / "stereo" / "depth_maps"
        dm_count = len(list(dm_dir.glob("*.photometric.bin"))) if dm_dir.exists() else 0
        print(f"1. Depth maps generated: {dm_count}")
        
        # 2. Normal maps count
        nm_dir = workspace / "dense" / "stereo" / "normal_maps"
        nm_count = len(list(nm_dir.glob("*.photometric.bin"))) if nm_dir.exists() else 0
        print(f"2. Normal maps generated: {nm_count}")
        
        # 3. Mesh vertex/face count
        import trimesh
        mesh = trimesh.load(results["mesh_path"])
        if isinstance(mesh, trimesh.Scene): mesh = mesh.dump(concatenate=True)
        print(f"3. Raw Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # 4. Connected components count
        # (This is available in cleanup_stats if we log it there, or we re-calculate)
        print(f"4. Connected components: {cleanup_stats.get('isolation', {}).get('component_count', 'N/A')}")
        
        # 5. Selected component reason (based on scores)
        score = cleanup_stats.get('isolation', {}).get('selected_component_score', 'N/A')
        print(f"5. Selected component score: {score}")
        
        # 6. GLB file size
        print(f"6. Final GLB size: {glb_path.stat().st_size / 1024:.2f} KB")
        
        # 7. Background geometry check (manual/visual logic placeholder)
        removed_planes = cleanup_stats.get('isolation', {}).get('removed_planes', 0)
        print(f"7. Removed planes: {removed_planes}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Fast Smoke Test
    success = run_test("Fast Smoke", num_images=10, max_image_size=800)
    
    if success:
        # Quality Test
        run_test("Quality Test", num_images=47, max_image_size=1600)
    else:
        print("Skipping Quality Test due to Fast Smoke failure.")
