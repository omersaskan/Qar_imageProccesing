import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add root to sys.path
ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT))

from modules.operations.settings import settings
from modules.reconstruction_engine.adapter import ColmapCommandBuilder
from modules.utils.file_persistence import atomic_write_json

def get_mesh_vertex_count(ply_path: Path) -> int:
    """Quickly read vertex count from PLY header."""
    if not ply_path.exists():
        return 0
    try:
        with open(ply_path, 'rb') as f:
            for line in f:
                if line.startswith(b'element vertex'):
                    return int(line.split()[-1])
                if line.startswith(b'end_header'):
                    break
    except Exception:
        pass
    return 0

def run_fusion_variant(
    variant_id: str,
    workspace: Path,
    colmap_bin: str,
    output_dir: Path,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    print(f"\n>>> Running Variant: {variant_id}")
    
    dense_dir = workspace / "dense"
    if not dense_dir.exists():
        return {"variant": variant_id, "status": "failed", "error": f"Dense directory not found: {dense_dir}"}

    images_dir = dense_dir / "images"
    image_count = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    mask_path = params.get("mask_path")
    if mask_path:
        mask_path = Path(mask_path)
        if not mask_path.is_absolute():
            mask_path = workspace / mask_path
    
    mask_exists = mask_path.exists() if mask_path else False
    mask_count = len(list(mask_path.glob("*.png"))) if mask_exists else 0
    
    # Filename match check
    matches = 0
    if mask_exists:
        image_filenames = {p.name for p in (list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))}
        mask_filenames = {p.name.replace(".png", "") for p in mask_path.glob("*.png")}
        matches = len(image_filenames.intersection(mask_filenames))

    output_ply = output_dir / f"fused_{variant_id}.ply"
    
    builder = ColmapCommandBuilder(colmap_bin)
    cmd = builder.stereo_fusion(
        dense_dir,
        output_ply,
        mask_path=str(mask_path) if mask_exists else None,
        min_num_pixels=params.get("min_num_pixels", settings.recon_stereo_fusion_min_num_pixels),
        max_reproj_error=params.get("max_reproj_error", settings.recon_stereo_fusion_max_reproj_error),
        max_depth_error=params.get("max_depth_error", settings.recon_stereo_fusion_max_depth_error),
        max_normal_error=params.get("max_normal_error", settings.recon_stereo_fusion_max_normal_error)
    )

    start_time = time.time()
    try:
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        runtime = time.time() - start_time
        
        status = "success" if result.returncode == 0 and output_ply.exists() else "failed"
        fused_count = get_mesh_vertex_count(output_ply)
        
        report = {
            "variant": variant_id,
            "status": status,
            "command": " ".join(cmd),
            "mask_path": str(mask_path) if mask_path else None,
            "mask_exists": mask_exists,
            "image_count": image_count,
            "mask_count": mask_count,
            "filename_matches": matches,
            "fused_point_count": fused_count,
            "output_ply": str(output_ply),
            "runtime_sec": round(runtime, 2),
            "stdout_tail": result.stdout[-500:] if result.stdout else "",
            "stderr_tail": result.stderr[-500:] if result.stderr else "",
            "return_code": result.returncode
        }
        return report
    except Exception as e:
        return {
            "variant": variant_id,
            "status": "error",
            "error": str(e),
            "runtime_sec": round(time.time() - start_time, 2)
        }

def main():
    parser = argparse.ArgumentParser(description="COLMAP Stereo Fusion Ablation Tool")
    parser.add_argument("--workspace", required=True, help="Path to reconstruction workspace (containing 'dense')")
    parser.add_argument("--output-dir", required=True, help="Directory to save ablation reports and PLYs")
    parser.add_argument("--variants", default="A1,A2,A3,C1", help="Comma-separated variants to run")
    parser.add_argument("--colmap-bin", default=settings.colmap_path, help="Path to COLMAP binary")

    args = parser.parse_args()
    workspace = Path(args.workspace)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    variants_to_run = args.variants.split(",")
    
    configs = {
        "A1": {"mask_path": None},
        "A2": {"mask_path": "masks"},
        "A3": {"mask_path": "dense/stereo/masks"},
        "C1": {
            "mask_path": "dense/stereo/masks",
            "max_reproj_error": 4.0,
            "max_depth_error": 0.03,
            "max_normal_error": 25.0,
            "min_num_pixels": 1
        }
    }

    full_report = {
        "workspace": str(workspace),
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "variants": []
    }

    for vid in variants_to_run:
        if vid not in configs:
            print(f"Skipping unknown variant: {vid}")
            continue
            
        res = run_fusion_variant(vid, workspace, args.colmap_bin, output_dir, configs[vid])
        full_report["variants"].append(res)
        
        # Immediate save
        atomic_write_json(output_dir / "ablation_report.json", full_report)

    print("\n\n" + "="*40)
    print("ABLATION SUMMARY")
    print("="*40)
    for v in full_report["variants"]:
        print(f"Variant {v['variant']}: {v['status']} | Points: {v.get('fused_point_count', 0)} | Matches: {v.get('filename_matches', 0)}/{v.get('image_count', 0)}")
    
    print(f"\nFull report saved to: {output_dir / 'ablation_report.json'}")

if __name__ == "__main__":
    main()
