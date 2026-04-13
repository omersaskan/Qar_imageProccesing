import subprocess
import os
import shutil
from pathlib import Path

COLMAP = r"C:\colmap\colmap\COLMAP.bat"
WORKSPACE = Path(r"c:\modelPlate\data\reconstructions\job_cap_24b4136c\attempt_1_denser_frames")
IMAGES_DIR = WORKSPACE / "images"
MASKS_DIR = WORKSPACE / "masks"
SPARSE_MODEL = WORKSPACE / "sparse" / "0"

VERIFY_DIR = Path(r"c:\modelPlate\data\reconstructions\full_verify_cap_24b4136c")
TEMP_MASKS_INPUT = VERIFY_DIR / "temp_masks_input"
DENSE_DIR = VERIFY_DIR / "dense"

def run_cmd(args, timeout=1200):
    print(f"\n> Running: {' '.join(args)}")
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
    for line in iter(proc.stdout.readline, ""):
        print(line.strip())
    proc.wait(timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}")

def run_full_rerun():
    if VERIFY_DIR.exists():
        shutil.rmtree(VERIFY_DIR)
    VERIFY_DIR.mkdir(parents=True)
    DENSE_DIR.mkdir(parents=True)
    TEMP_MASKS_INPUT.mkdir(parents=True)

    print(f"Starting Intelligent Full Rerun for cap_24b4136c...")
    
    # 1. Undistort IMAGES
    run_cmd([
        COLMAP, "image_undistorter",
        "--image_path", str(IMAGES_DIR),
        "--input_path", str(SPARSE_MODEL),
        "--output_path", str(DENSE_DIR),
        "--output_type", "COLMAP"
    ])

    # 2. Undistort MASKS (via temporary swap)
    print("\n[Step] Undistorting masks via temp swap...")
    # Prepare temp images dir with masks renamed to .jpg
    for m in MASKS_DIR.glob("*.png"):
        # Match frame name: frame_0000.jpg.png -> frame_0000.jpg
        target_name = m.name.replace(".png", "")
        shutil.copy2(m, TEMP_MASKS_INPUT / target_name)
    
    TEMP_DENSE_MASKS = VERIFY_DIR / "temp_dense_masks"
    TEMP_DENSE_MASKS.mkdir()
    
    run_cmd([
        COLMAP, "image_undistorter",
        "--image_path", str(TEMP_MASKS_INPUT),
        "--input_path", str(SPARSE_MODEL),
        "--output_path", str(TEMP_DENSE_MASKS),
        "--output_type", "COLMAP"
    ])
    
    # Move undistorted masks to DENSE_DIR/masks
    target_masks_dir = DENSE_DIR / "masks"
    target_masks_dir.mkdir()
    for m in (TEMP_DENSE_MASKS / "images").glob("*.jpg"):
        # Rename back: frame_0000.jpg -> frame_0000.jpg.png
        shutil.move(m, target_masks_dir / f"{m.name}.png")
    
    print(f"Undistorted {len(list(target_masks_dir.glob('*.png')))} masks successfully.")

    # 3. PatchMatch Stereo
    # Now it HAS a masks folder in the workspace!
    run_cmd([
        COLMAP, "patch_match_stereo",
        "--workspace_path", str(DENSE_DIR),
        "--PatchMatchStereo.gpu_index", "0",
        "--PatchMatchStereo.geom_consistency", "1",
        "--PatchMatchStereo.filter", "1"
    ])

    # 4. Stereo Fusion
    fused_ply = DENSE_DIR / "fused.ply"
    run_cmd([
        COLMAP, "stereo_fusion",
        "--workspace_path", str(DENSE_DIR),
        "--output_path", str(fused_ply)
    ])

    # 5. Poisson Mesher
    mesh_ply = DENSE_DIR / "meshed-poisson.ply"
    run_cmd([
        COLMAP, "poisson_mesher",
        "--input_path", str(fused_ply),
        "--output_path", str(mesh_ply)
    ])

    print("\n--- Final Artifacts ---")
    for f in DENSE_DIR.glob("*.ply"):
        print(f"{f.name}: {f.stat().st_size} bytes")

if __name__ == "__main__":
    try:
        run_full_rerun()
    except Exception as e:
        print(f"\nFATAL: {e}")
