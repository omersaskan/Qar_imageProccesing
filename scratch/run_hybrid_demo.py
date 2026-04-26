import os
import json
from pathlib import Path
from modules.reconstruction_engine.adapter import COLMAPAdapter
from modules.operations.settings import settings

def run_demo():
    input_dir = Path("scratch/demo_hybrid/images").absolute()
    output_dir = Path("scratch/demo_hybrid/output").absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3 dummy frames
    frames = [str(f) for f in input_dir.glob("*.jpg")]
    
    # Enable hybrid masking
    settings.recon_hybrid_masking = True
    
    # We expect SfM to likely fail on black images, but we want to see the initialization logs
    adapter = COLMAPAdapter(settings.colmap_path)
    
    # Mock _run_command to let it "finish" even with dummy images
    original_run = adapter._run_command
    def mock_run(cmd, output_dir, log_file, timeout=None):
        print(f"Mocking command: {cmd[1]}")
        # Create a dummy sparse model so select_best_sparse_model doesn't fail
        sparse_dir = output_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        (sparse_dir / "cameras.bin").touch()
        (sparse_dir / "images.bin").touch()
        (sparse_dir / "points3D.bin").touch()
        return 0

    import shutil
    from unittest.mock import patch
    
    with patch.object(COLMAPAdapter, "_run_command", side_effect=mock_run), \
         patch.object(COLMAPAdapter, "_select_best_sparse_model", return_value={"path": output_dir / "sparse" / "0", "registered_images": 5, "points_3d": 1000}), \
         patch.object(COLMAPAdapter, "_validate_dense_workspace", return_value=1000), \
         patch.object(COLMAPAdapter, "_discover_mesh_candidates", return_value=[str(output_dir / "dense" / "fused.ply")]), \
         patch.object(COLMAPAdapter, "_is_valid_mesh_candidate", return_value=True):
        
        # Create dummy fused.ply so discovery works
        dense_dir = output_dir / "dense"
        dense_dir.mkdir(parents=True, exist_ok=True)
        (dense_dir / "fused.ply").touch()
        
        # Create dummy dense images so mask generation doesn't skip
        dense_images_dir = dense_dir / "images"
        dense_images_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            (dense_images_dir / f"f{i}.jpg").touch()

        print(f"Running demo reconstruction for {len(frames)} frames...")
        result = adapter.run_reconstruction(
            input_frames=frames,
            output_dir=output_dir,
            enforce_masks=True
        )
        print("Reconstruction finished.")
        print(json.dumps(result, indent=2))
    
    # Evidence 1: reconstruction.log
    log_path = output_dir / "reconstruction.log"
    if log_path.exists():
        print("\n--- RECONSTRUCTION.LOG ---")
        print(log_path.read_text(encoding='utf-8'))
    
    # Evidence 2: fusion_diagnostics.json
    diag_path = output_dir / "fusion_diagnostics.json"
    if diag_path.exists():
        print("\n--- FUSION_DIAGNOSTICS.JSON ---")
        print(diag_path.read_text(encoding='utf-8'))

if __name__ == "__main__":
    run_demo()
