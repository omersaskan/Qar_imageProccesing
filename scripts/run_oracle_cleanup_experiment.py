
import os
import cv2
import numpy as np
from pathlib import Path
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.camera_projection import load_reconstruction_cameras
from modules.operations.settings import settings

def run_experiment():
    job_id = "cap_29ab6fa1"
    recon_dir = Path(f"data/reconstructions/job_{job_id}/attempt_0_default")
    raw_mesh = recon_dir / "mesh.obj"
    raw_texture = recon_dir / "texture.jpg"
    
    # Fallback
    if not raw_mesh.exists():
        raw_mesh = Path(f"data/cleaned/job_{job_id}/pre_aligned_mesh.obj")
        raw_texture = Path(f"data/cleaned/job_{job_id}/textured_model_material_00_map_Kd.jpg")

    if not raw_mesh.exists():
        print(f"Error: Could not find mesh for {job_id}")
        return

    # Load Cameras
    cameras = load_reconstruction_cameras(recon_dir)
    print(f"Loaded {len(cameras)} cameras")

    # Load Oracle masks
    mask_dir = Path(f"data/captures/{job_id}/frames/oracle_masks")
    masks = {}
    for m_path in mask_dir.glob("*.png"):
        # Map frame_0000.png to the camera name if possible, or just use the name
        # The cameras usually have names like 'frame_0000.jpg'
        masks[m_path.stem + ".jpg"] = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        
    print(f"Loaded {len(masks)} Oracle masks")
    
    cleaner = AssetCleaner()
    
    # Move SAM2 masks to 'official' location for metadata detection
    official_masks_dir = Path(f"data/captures/{job_id}/frames/masks")
    backup_masks_dir = Path(f"data/captures/{job_id}/frames/masks_legacy")
    if not backup_masks_dir.exists():
        official_masks_dir.rename(backup_masks_dir)
        mask_dir.rename(official_masks_dir)
    
    try:
        # Run cleanup
        metadata, stats, mesh_path = cleaner.process_cleanup(
            job_id=f"{job_id}_sam2_experiment",
            raw_mesh_path=str(raw_mesh),
            raw_texture_path=str(raw_texture),
            cameras=cameras,
            masks=masks
        )
        
        print("\n--- SAM2 Experiment Results ---")
        print(f"Status: {stats.get('delivery_ready')}")
        print(f"Isolation Method: {stats['isolation']['object_isolation_method']}")
        print(f"Isolation Confidence: {stats['isolation']['isolation_confidence']:.2f}")
        print(f"Used SAM2: {stats['isolation'].get('used_sam2')}")
        print(f"Mask Support Ratio: {stats['isolation'].get('mask_support_ratio', 0.0):.2f}")
        
        # Save experiment results
        with open("sam2_experiment_results.json", "w") as f:
            import json
            json.dump(stats, f, indent=2)
            
    finally:
        # Restore legacy masks
        if backup_masks_dir.exists():
            official_masks_dir.rename(mask_dir)
            backup_masks_dir.rename(official_masks_dir)

if __name__ == "__main__":
    run_experiment()
