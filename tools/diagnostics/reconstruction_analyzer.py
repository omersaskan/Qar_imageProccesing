import argparse
import sys
from pathlib import Path
from modules.operations.settings import settings
from modules.reconstruction_engine.adapter import ColmapCommandBuilder

def main():
    parser = argparse.ArgumentParser(description="Diagnostic tool for reconstruction session analysis.")
    parser.add_argument("--session-id", required=True, help="ID of the session to analyze (e.g. cap_123)")
    parser.add_argument("--data-root", help="Override standard data root (default: settings.data_root)")
    
    args = parser.parse_args()
    
    # 1. Resolve paths
    data_root = Path(args.data_root or settings.data_root).resolve()
    captures_dir = data_root / "captures"
    session_dir = captures_dir / args.session_id
    
    output_root = data_root / "reconstructions"
    job_dir = output_root / f"job_{args.session_id}"
    
    # User Request: resolve and print immediately
    print("--- Path Resolution ---")
    print(f"Resolved Data Root:    {data_root}")
    print(f"Resolved Captures:     {captures_dir}")
    print(f"Resolved Session Dir:  {session_dir}")
    print(f"Recon Job Directory:   {job_dir}")
    print("-" * 23)

    if not session_dir.exists():
        print(f"ERROR: Session directory does not exist: {session_dir}")
        sys.exit(1)

    # 2. Audit Logs
    log_file = job_dir / "reconstruction.log"
    if not log_file.exists():
        print(f"Warning: Reconstruction log missing at {log_file}")
    else:
        print(f"Found log at {log_file}. Analyzing...")
        with open(log_file, "r") as f:
            lines = f.readlines()
            # Look for sparse model stats
            for line in lines:
                if "Sparse model stats" in line:
                    print(f"MATCH: {line.strip()}")
                if "Fused points" in line:
                    print(f"MATCH: {line.strip()}")
                if "Delaunay mesher failed" in line:
                    print(f"FAILURE: {line.strip()}")

    # 3. Audit Artifacts
    dense_dir = job_dir / "dense"
    fused_ply = dense_dir / "fused.ply"
    if fused_ply.exists():
        size = fused_ply.stat().st_size
        print(f"Artifact: fused.ply exists ({size} bytes)")
    else:
        print("Artifact: fused.ply MISSING")

    # 4. Verify CLI Builder (dry run)
    builder = ColmapCommandBuilder(settings.colmap_path)
    print("\n--- Command Gating Check ---")
    delaunay_cmd = builder.delaunay_mesher(dense_dir, dense_dir / "test_mesh.ply")
    print(f"Delaunay Fix check: {delaunay_cmd}")
    if str(dense_dir / "fused.ply") in " ".join(delaunay_cmd):
        print("FAIL: Delaunay command still points directly to fused.ply")
    else:
        print("PASS: Delaunay command correctly points to workspace root")

if __name__ == "__main__":
    main()
