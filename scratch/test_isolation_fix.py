import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.camera_projection import load_reconstruction_cameras, load_reconstruction_masks
from modules.reconstruction_engine.output_manifest import OutputManifest
import trimesh

def test_cleanup_on_existing_job():
    job_id = "cap_29ab6fa1"
    
    # Load manifest
    manifest_path = Path(f"data/reconstructions/job_{job_id}/manifest.json")
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)
        manifest = OutputManifest(**manifest_data)
    
    # Resolve guidance data (like worker.py)
    workspace_path = Path(manifest.mesh_path).parent.parent
    cameras = load_reconstruction_cameras(workspace_path)
    masks = load_reconstruction_masks(workspace_path, [c["name"] for c in cameras])
    
    fused_path = workspace_path / "dense" / "fused.ply"
    point_cloud = trimesh.load(str(fused_path)) if fused_path.exists() else None
    
    print(f"Testing cleanup with: cameras={len(cameras)}, masks={len(masks)}, point_cloud={bool(point_cloud)}")
    
    cleaner = AssetCleaner()
    # We'll use a dummy job_id to not overwrite
    test_job_id = f"test_{job_id}"
    
    metadata, cleanup_stats, cleaned_mesh_path = cleaner.process_cleanup(
        job_id=test_job_id,
        raw_mesh_path=manifest.mesh_path,
        cameras=cameras,
        masks=masks,
        point_cloud=point_cloud
    )
    
    print(f"Isolation Method: {cleanup_stats['isolation'].get('object_isolation_method')}")
    print(f"Mask Support Ratio: {cleanup_stats['isolation'].get('mask_support_ratio')}")
    print(f"Point Cloud Support Ratio: {cleanup_stats['isolation'].get('point_cloud_support_ratio')}")
    print(f"Reason if Fallback: {cleanup_stats['isolation'].get('reason_if_geometric_fallback')}")

if __name__ == "__main__":
    test_cleanup_on_existing_job()
