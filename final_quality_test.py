import os
import sys
import shutil
from pathlib import Path
from modules.operations.settings import Settings
from modules.reconstruction_engine.adapter import COLMAPAdapter

def main():
    workspace = Path("workspace_final_quality").absolute()
    
    # 1. CLEAN START
    if workspace.exists():
        print(f"Cleaning up existing workspace: {workspace}")
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    
    # Prepare images and masks
    images_src = Path("workspace_quality_test/images")
    masks_src = Path("workspace_quality_test/masks")
    
    images_dst = workspace / "images"
    masks_dst = workspace / "masks"
    
    shutil.copytree(images_src, images_dst)
    shutil.copytree(masks_src, masks_dst)
    
    # 2. Configure Relaxed Settings
    settings = Settings()
    # Explicitly set the values to match the user request
    settings.recon_stereo_fusion_min_num_pixels = 1
    settings.recon_stereo_fusion_max_reproj_error = 4.0
    settings.recon_stereo_fusion_max_depth_error = 0.03
    settings.recon_stereo_fusion_max_normal_error = 25.0
    settings.recon_max_image_size = 1600
    
    adapter = COLMAPAdapter(settings_override=settings)
    
    print("--- Starting Final Quality Test ---")
    print(f"Pipeline: colmap_dense")
    print(f"Max Image Size: 1600")
    print(f"Min Num Pixels: {settings.recon_stereo_fusion_min_num_pixels}")
    print(f"Max Reproj Error: {settings.recon_stereo_fusion_max_reproj_error}")
    
    # Prepare frames list
    input_frames = [str(p) for p in images_src.glob("*.jpg")]
    if not input_frames:
        print(f"ERROR: No images found in {images_src}")
        sys.exit(1)
        
    print(f"Total images found: {len(input_frames)}")

    try:
        # 3. Run Reconstruction
        # Note: the adapter handles workspace preparation (copying images/masks)
        result = adapter.run_reconstruction(
            input_frames=input_frames,
            output_dir=workspace,
            density=1.0,
            enforce_masks=True
        )
        
        print("\n--- RECONSTRUCTION SUCCESSFUL ---")
        print(f"Model Path: {result['mesh_path']}")
        print(f"Fused Point Count: {result['dense_points_fused']}")
        print(f"Diagnostics: {result['diagnostics_path']}")
        
        # Write report
        with open("final_test_report.txt", "w") as f:
            f.write(f"Final Quality Test Report\n")
            f.write(f"=========================\n")
            f.write(f"Fused Points: {result['dense_points_fused']}\n")
            f.write(f"Mesh Path: {result['mesh_path']}\n")
            f.write(f"Diagnostics: {result['diagnostics_path']}\n")
            
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
