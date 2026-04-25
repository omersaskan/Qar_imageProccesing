import os
import sys
import shutil
from pathlib import Path
from modules.operations.settings import Settings
from modules.reconstruction_engine.adapter import COLMAPAdapter

def run_fusion_variant(name, adapter, workspace, mask_path, thresholds):
    print(f"\n--- Running Variant: {name} ---")
    
    # Create variant-specific output dir
    variant_dir = workspace / f"fusion_{name}"
    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    variant_dir.mkdir(parents=True)
    
    output_ply = variant_dir / "fused.ply"
    log_file_path = variant_dir / "fusion.log"
    
    # Build command
    # We bypass the adapter's orchestration to run JUST fusion
    # adapter.builder.stereo_fusion(...)
    cmd = adapter.builder.stereo_fusion(
        workspace_path=workspace / "dense",
        output_path=output_ply,
        mask_path=mask_path,
        **thresholds
    )
    
    print(f"Command: {' '.join(cmd)}")
    
    with open(log_file_path, "w") as log_file:
        adapter._run_command(cmd, workspace, log_file)
    
    # Analyze results
    stats = {"name": name, "point_count": 0, "ply_exists": output_ply.exists()}
    if output_ply.exists():
        # Get point count from PLY (hacky way: count lines starting with 'element vertex')
        with open(output_ply, "rb") as f:
            header = f.read(1024).decode(errors="ignore")
            for line in header.splitlines():
                if line.startswith("element vertex"):
                    stats["point_count"] = int(line.split()[-1])
                    break
    
    print(f"Result: {stats['point_count']} fused points")
    return stats

def main():
    # Use the existing workspace from the running test if it has depth maps
    # Or expect a prepared workspace
    workspace = Path("workspace_final_quality").absolute()
    if not (workspace / "dense/stereo/depth_maps").exists():
        print(f"Error: Workspace {workspace} does not have depth maps. Run reconstruction first.")
        sys.exit(1)
        
    settings = Settings()
    adapter = COLMAPAdapter(settings_override=settings)
    
    dense_masks = workspace / "dense/stereo/masks"
    raw_masks = workspace / "masks"
    
    variants = [
        {
            "name": "A1_NoMask_Default",
            "mask_path": None,
            "thresholds": {
                "min_num_pixels": 2,
                "max_reproj_error": 2.0,
                "max_depth_error": 0.01,
                "max_normal_error": 10.0
            }
        },
        {
            "name": "A2_RawMask_Default",
            "mask_path": raw_masks,
            "thresholds": {
                "min_num_pixels": 2,
                "max_reproj_error": 2.0,
                "max_depth_error": 0.01,
                "max_normal_error": 10.0
            }
        },
        {
            "name": "A3_DenseMask_Default",
            "mask_path": dense_masks,
            "thresholds": {
                "min_num_pixels": 2,
                "max_reproj_error": 2.0,
                "max_depth_error": 0.01,
                "max_normal_error": 10.0
            }
        },
        {
            "name": "C1_DenseMask_Relaxed",
            "mask_path": dense_masks,
            "thresholds": {
                "min_num_pixels": 1,
                "max_reproj_error": 4.0,
                "max_depth_error": 0.03,
                "max_normal_error": 25.0
            }
        }
    ]
    
    results = []
    for var in variants:
        try:
            res = run_fusion_variant(
                var["name"], 
                adapter, 
                workspace, 
                var["mask_path"], 
                var["thresholds"]
            )
            results.append(res)
        except Exception as e:
            print(f"Variant {var['name']} failed: {e}")
            
    # Print Summary Table
    print("\n--- A/B/C Ablation Summary ---")
    print(f"{'Variant':<25} | {'Points':<10} | {'PLY Status'}")
    print("-" * 50)
    for r in results:
        status = "OK" if r["ply_exists"] else "MISSING"
        print(f"{r['name']:<25} | {r['point_count']:<10} | {status}")

if __name__ == "__main__":
    main()
