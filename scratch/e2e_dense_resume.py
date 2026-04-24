"""
Resume from patch_match_stereo onward.
image_undistorter already completed successfully (29 undistorted images).
"""
import subprocess
from pathlib import Path

COLMAP = r"C:\colmap\colmap\COLMAP.bat"
DENSE_DIR = Path(r"c:\modelPlate\data\reconstructions\job_cap_24b4136c\attempt_1_denser_frames\dense")
LOG_PATH = DENSE_DIR.parent / "e2e_dense_rerun.log"


def run_step(name, cmd, log_file, timeout=600):
    header = f"\n{'='*60}\n[STEP] {name}\n{'='*60}\n"
    print(header, end="", flush=True)
    log_file.write(header)

    cmd_str = " ".join(cmd)
    print(f"CMD: {cmd_str}\n", flush=True)
    log_file.write(f"CMD: {cmd_str}\n")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, shell=True
    )
    stdout, _ = proc.communicate(timeout=timeout)
    log_file.write(stdout)
    log_file.flush()

    lines = stdout.strip().splitlines()
    for line in lines[-20:]:
        print(line, flush=True)

    if proc.returncode != 0:
        msg = f"\nFAILED: {name} exited with code {proc.returncode}"
        print(msg, flush=True)
        log_file.write(msg + "\n")
        return False

    msg = f"\nOK: {name} completed successfully."
    print(msg, flush=True)
    log_file.write(msg + "\n")
    return True


def main():
    with open(LOG_PATH, "w", encoding="utf-8") as log:
        log.write("E2E Dense Rerun (resumed from patch_match_stereo)\n")
        log.write(f"Dense workspace: {DENSE_DIR}\n\n")

        # image_undistorter already done — 29 images in dense/images/
        log.write("[SKIP] image_undistorter (already completed)\n")
        print("[SKIP] image_undistorter (already completed)", flush=True)

        # Step 2: patch_match_stereo (GPU heavy, needs longer timeout)
        ok = run_step("patch_match_stereo", [
            COLMAP, "patch_match_stereo",
            "--workspace_path", str(DENSE_DIR),
            "--PatchMatchStereo.gpu_index", "0",
            "--PatchMatchStereo.geom_consistency", "1",
            "--PatchMatchStereo.filter", "1",
        ], log, timeout=600)
        if not ok:
            print("\nBLOCKER: patch_match_stereo failed.", flush=True)
            return

        # Step 3: stereo_fusion
        fused_ply = DENSE_DIR / "fused.ply"
        ok = run_step("stereo_fusion", [
            COLMAP, "stereo_fusion",
            "--workspace_path", str(DENSE_DIR),
            "--output_path", str(fused_ply),
        ], log, timeout=300)
        if not ok:
            print("\nBLOCKER: stereo_fusion failed.", flush=True)
            return

        if fused_ply.exists():
            size = fused_ply.stat().st_size
            print(f"\nfused.ply: {size} bytes", flush=True)
            log.write(f"\nfused.ply size: {size} bytes\n")
        else:
            print("\nBLOCKER: fused.ply missing.", flush=True)
            return

        # Step 4: poisson_mesher
        poisson_mesh = DENSE_DIR / "meshed-poisson.ply"
        ok = run_step("poisson_mesher", [
            COLMAP, "poisson_mesher",
            "--input_path", str(fused_ply),
            "--output_path", str(poisson_mesh),
        ], log, timeout=300)

        if ok and poisson_mesh.exists():
            size = poisson_mesh.stat().st_size
            print(f"\nmeshed-poisson.ply: {size} bytes", flush=True)
            log.write(f"\nmeshed-poisson.ply size: {size} bytes\n")
        else:
            print("\nPoisson failed, trying delaunay...", flush=True)
            delaunay_mesh = DENSE_DIR / "meshed-delaunay.ply"
            ok = run_step("delaunay_mesher", [
                COLMAP, "delaunay_mesher",
                "--input_path", str(DENSE_DIR),
                "--output_path", str(delaunay_mesh),
            ], log, timeout=300)
            if ok and delaunay_mesh.exists():
                size = delaunay_mesh.stat().st_size
                print(f"\nmeshed-delaunay.ply: {size} bytes", flush=True)
            else:
                print("\nBLOCKER: Both meshers failed.", flush=True)
                return

        # Summary
        print(f"\n{'='*60}", flush=True)
        print("ALL DENSE STAGES COMPLETED SUCCESSFULLY", flush=True)
        artifacts = [f.name for f in DENSE_DIR.iterdir() if f.is_file()]
        print(f"Artifacts: {artifacts}", flush=True)
        log.write(f"\nALL STAGES COMPLETE. Artifacts: {artifacts}\n")


if __name__ == "__main__":
    main()
