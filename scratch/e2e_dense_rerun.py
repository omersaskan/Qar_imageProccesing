"""
End-to-end dense reconstruction rerun for cap_24b4136c.

Uses the existing valid sparse model (29 images, 4187 points) and
runs the real COLMAP binaries for the stages beyond the sparse gate:
  1. image_undistorter
  2. patch_match_stereo
  3. stereo_fusion
  4. poisson_mesher (fallback: delaunay_mesher)

This is NOT a mock or dry-run — it invokes the actual COLMAP.bat.
"""

import subprocess
import sys
import os
from pathlib import Path

COLMAP = r"C:\colmap\colmap\COLMAP.bat"
WORKSPACE = Path(r"c:\modelPlate\data\reconstructions\job_cap_24b4136c\attempt_1_denser_frames")
IMAGES_DIR = WORKSPACE / "images"
SPARSE_MODEL = WORKSPACE / "sparse" / "0"
DENSE_DIR = WORKSPACE / "dense"

LOG_PATH = WORKSPACE / "e2e_dense_rerun.log"


def run_step(name: str, cmd: list, log_file):
    """Run a COLMAP step and stream output to log and stdout."""
    header = f"\n{'='*60}\n[STEP] {name}\n{'='*60}\n"
    print(header, end="")
    log_file.write(header)

    cmd_str = " ".join(cmd)
    print(f"CMD: {cmd_str}\n")
    log_file.write(f"CMD: {cmd_str}\n")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, shell=True
    )
    stdout, _ = proc.communicate(timeout=600)
    log_file.write(stdout)

    # Print last 15 lines to console
    lines = stdout.strip().splitlines()
    for line in lines[-15:]:
        print(line)

    if proc.returncode != 0:
        msg = f"\nFAILED: {name} exited with code {proc.returncode}"
        print(msg)
        log_file.write(msg + "\n")
        return False

    msg = f"\nOK: {name} completed successfully."
    print(msg)
    log_file.write(msg + "\n")
    return True


def main():
    # Clean previous dense output if exists
    if DENSE_DIR.exists():
        import shutil
        shutil.rmtree(DENSE_DIR)
    DENSE_DIR.mkdir(parents=True)

    with open(LOG_PATH, "w", encoding="utf-8") as log:
        log.write(f"E2E Dense Rerun for cap_24b4136c\n")
        log.write(f"Sparse model: {SPARSE_MODEL}\n")
        log.write(f"Images: {IMAGES_DIR}\n")
        log.write(f"Dense output: {DENSE_DIR}\n\n")

        # Step 1: image_undistorter
        ok = run_step("image_undistorter", [
            COLMAP, "image_undistorter",
            "--image_path", str(IMAGES_DIR),
            "--input_path", str(SPARSE_MODEL),
            "--output_path", str(DENSE_DIR),
            "--output_type", "COLMAP",
        ], log)
        if not ok:
            print("\nBLOCKER: image_undistorter failed. Cannot proceed to dense.")
            return

        # Step 2: patch_match_stereo
        ok = run_step("patch_match_stereo", [
            COLMAP, "patch_match_stereo",
            "--workspace_path", str(DENSE_DIR),
            "--PatchMatchStereo.gpu_index", "0",
            "--PatchMatchStereo.geom_consistency", "1",
            "--PatchMatchStereo.filter", "1",
        ], log)
        if not ok:
            print("\nBLOCKER: patch_match_stereo failed.")
            return

        # Step 3: stereo_fusion
        fused_ply = DENSE_DIR / "fused.ply"
        ok = run_step("stereo_fusion", [
            COLMAP, "stereo_fusion",
            "--workspace_path", str(DENSE_DIR),
            "--output_path", str(fused_ply),
        ], log)
        if not ok:
            print("\nBLOCKER: stereo_fusion failed.")
            return

        # Validate fused.ply
        if fused_ply.exists():
            size = fused_ply.stat().st_size
            log.write(f"\nfused.ply size: {size} bytes\n")
            print(f"\nfused.ply written: {size} bytes")
        else:
            print("\nBLOCKER: fused.ply was not created.")
            return

        # Step 4: Meshing (poisson first, then delaunay fallback)
        poisson_mesh = DENSE_DIR / "meshed-poisson.ply"
        ok = run_step("poisson_mesher", [
            COLMAP, "poisson_mesher",
            "--input_path", str(fused_ply),
            "--output_path", str(poisson_mesh),
        ], log)

        if ok and poisson_mesh.exists():
            size = poisson_mesh.stat().st_size
            print(f"\nmeshed-poisson.ply written: {size} bytes")
            log.write(f"\nmeshed-poisson.ply size: {size} bytes\n")
        else:
            print("\nPoisson failed, trying delaunay...")
            delaunay_mesh = DENSE_DIR / "meshed-delaunay.ply"
            ok = run_step("delaunay_mesher", [
                COLMAP, "delaunay_mesher",
                "--input_path", str(DENSE_DIR),
                "--output_path", str(delaunay_mesh),
            ], log)
            if ok and delaunay_mesh.exists():
                size = delaunay_mesh.stat().st_size
                print(f"\nmeshed-delaunay.ply written: {size} bytes")
                log.write(f"\nmeshed-delaunay.ply size: {size} bytes\n")
            else:
                print("\nBLOCKER: Both meshers failed.")
                return

        # Final summary
        summary = "\n" + "="*60 + "\nFINAL SUMMARY\n" + "="*60
        summary += f"\nAll dense stages completed successfully."
        summary += f"\nArtifacts in: {DENSE_DIR}"
        artifacts = list(DENSE_DIR.glob("*"))
        summary += f"\nFiles: {[a.name for a in artifacts if a.is_file()]}"
        print(summary)
        log.write(summary + "\n")


if __name__ == "__main__":
    main()
