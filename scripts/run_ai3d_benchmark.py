import os
import sys
import json
import csv
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from modules.ai_3d_generation.pipeline import generate_ai_3d
from modules.operations.settings import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ai3d_benchmark")

def get_mesh_stats(glb_path):
    stats = {
        "vertex_count": 0,
        "face_count": 0,
        "geometry_count": 0,
        "mesh_stats_available": False
    }
    if not glb_path or not os.path.exists(glb_path):
        return stats
    
    try:
        import trimesh
        scene = trimesh.load(glb_path, force='scene')
        if isinstance(scene, trimesh.Scene):
            stats["geometry_count"] = len(scene.geometry)
            for mesh in scene.geometry.values():
                if hasattr(mesh, 'vertices'):
                    stats["vertex_count"] += len(mesh.vertices)
                if hasattr(mesh, 'faces'):
                    stats["face_count"] += len(mesh.faces)
        else:
            # Single mesh
            stats["geometry_count"] = 1
            stats["vertex_count"] = len(scene.vertices)
            stats["face_count"] = len(scene.faces)
            
        stats["mesh_stats_available"] = True
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to extract mesh stats from {glb_path}: {e}")
        
    return stats

def run_benchmark():
    parser = argparse.ArgumentParser(description="SF3D Local Benchmark Runner")
    parser.add_argument("--input-dir", default="scratch/ai3d_benchmark_inputs", help="Directory with input images")
    parser.add_argument("--output-dir", default="reports/ai3d_benchmark", help="Directory for report files")
    parser.add_argument("--modes", default="balanced,high,ultra", help="Comma-separated quality modes")
    parser.add_argument("--bg-modes", default="off,on", help="Comma-separated background removal modes (off/on)")
    parser.add_argument("--limit", type=int, help="Limit number of inputs to process")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    modes = args.modes.split(",")
    bg_enabled_options = [m.lower() == "on" for m in args.bg_modes.split(",")]
    
    inputs = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if args.limit:
        inputs = inputs[:args.limit]
        
    results = []
    
    logger.info(f"Starting benchmark: {len(inputs)} inputs, {len(modes)} modes, {len(bg_enabled_options)} bg-modes")
    
    # Ensure AI 3D is enabled for benchmark
    settings.ai_3d_generation_enabled = True
    
    for input_file in inputs:
        for mode in modes:
            for bg_removal in bg_enabled_options:
                benchmark_id = f"bench_{datetime.now(tz=timezone.utc).strftime('%H%M%S')}_{input_file.stem}_{mode}_bg{'on' if bg_removal else 'off'}"
                logger.info(f"--- Running {benchmark_id} ---")
                
                try:
                    # Create session dir
                    session_id = f"bench_{datetime.now(tz=timezone.utc).strftime('%H%M%S')}_{input_file.stem}"
                    session_dir = output_dir / session_id
                    session_dir.mkdir(parents=True, exist_ok=True)

                    # Run generation
                    manifest = generate_ai_3d(
                        session_id=session_id,
                        input_file_path=str(input_file),
                        output_base_dir=str(session_dir),
                        provider_name="sf3d",
                        options={
                            "quality_mode": mode,
                            "background_removal_enabled": bg_removal
                        }
                    )
                    
                    # Collect metrics
                    status = manifest.get("status")
                    provider_status = manifest.get("provider_status")
                    duration = manifest.get("duration_sec", 0)
                    glb_path = manifest.get("output_glb_path")
                    glb_size = manifest.get("output_size_bytes") or 0
                    peak_mem = manifest.get("peak_mem_mb") or 0
                    
                    pre = manifest.get("preprocessing", {})
                    mesh_stats = get_mesh_stats(glb_path)
                    
                    row = {
                        "benchmark_id": benchmark_id,
                        "input_filename": input_file.name,
                        "quality_mode": mode,
                        "background_removal_enabled": bg_removal,
                        "session_id": manifest.get("session_id"),
                        "status": status,
                        "provider_status": provider_status,
                        "input_mode": manifest.get("input_mode"),
                        "candidate_count": manifest.get("candidate_count"),
                        "selected_candidate_id": manifest.get("selected_candidate_id"),
                        "duration_sec": duration,
                        "output_size_bytes": glb_size,
                        "peak_mem_mb": peak_mem,
                        "device": manifest.get("worker_metadata", {}).get("device"),
                        "input_size": manifest.get("resolved_quality", {}).get("input_size"),
                        "bg_removed": pre.get("background_removed"),
                        "mask_source": pre.get("mask_source"),
                        "foreground_ratio": pre.get("foreground_ratio_estimate"),
                        "score": manifest.get("candidate_ranking", [{}])[0].get("score") if manifest.get("candidate_ranking") else None,
                        "vertex_count": mesh_stats["vertex_count"],
                        "face_count": mesh_stats["face_count"],
                        "mesh_stats_available": mesh_stats["mesh_stats_available"],
                        "output_glb_path": glb_path,
                        "prepared_image_path": manifest.get("prepared_image_path"),
                        "warnings_count": len(manifest.get("warnings", [])),
                        "errors_count": len(manifest.get("errors", []))
                    }
                    results.append(row)
                    logger.info(f"Result: {status} in {duration}s, GLB: {glb_size} bytes")
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {benchmark_id}: {e}")
                    results.append({
                        "benchmark_id": benchmark_id,
                        "input_filename": input_file.name,
                        "status": "failed",
                        "error": str(e)
                    })

    # Save JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
        
    # Save CSV
    csv_path = output_dir / "results.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
    # Save Markdown Report
    md_path = output_dir / "AI_3D_PHASE3A_SF3D_BENCHMARK_REPORT.md"
    success_count = len([r for r in results if r.get("status") == "review"])
    
    with open(md_path, "w") as f:
        f.write("# AI 3D Phase 3A — SF3D Local Benchmark Report\n\n")
        f.write(f"- **Date**: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        # In a real environment I'd get the git SHA here, but I'll placeholder it for now
        f.write(f"- **Environment**: Local Windows/WSL2\n")
        f.write(f"- **Total Inputs**: {len(inputs)}\n")
        f.write(f"- **Total Runs**: {len(results)}\n")
        f.write(f"- **Successful Runs**: {success_count}\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Input | Mode | BG | Status | Duration | GLB Size | Peak VRAM | Score |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            if "error" in r:
                f.write(f"| {r['input_filename']} | - | - | FAILED | - | - | - | - |\n")
                continue
            bg_str = "ON" if r["background_removal_enabled"] else "OFF"
            size_mb = round(r["output_size_bytes"] / (1024*1024), 2)
            f.write(f"| {r['input_filename']} | {r['quality_mode']} | {bg_str} | {r['status']} | {r['duration_sec']}s | {size_mb} MB | {r['peak_mem_mb']} MB | {r['score']} |\n")
            
        f.write("\n## Notes\n\n")
        f.write("- This benchmark covers only local SF3D.\n")
        f.write("- External providers remain disabled and were not touched.\n")
        f.write("- This is not true multi-view reconstruction.\n")
        f.write("- Mesh statistics (vertex/face) are collected via `trimesh` if available.\n")

    logger.info(f"Benchmark complete. Reports saved to {output_dir}")

if __name__ == "__main__":
    run_benchmark()
