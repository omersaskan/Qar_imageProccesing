import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import trimesh
from modules.reconstruction_engine.adapter import COLMAPAdapter
from modules.reconstruction_engine.failures import InsufficientInputError


def write_frame_and_mask(frame_path: Path):
    frame_path.write_bytes(b"frame")

    masks_dir = frame_path.parent / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    (masks_dir / f"{frame_path.name}.png").write_bytes(b"mask")

def test_colmap_adapter_init():
    os.environ["RECON_ENGINE_PATH"] = "C:/colmap/COLMAP.bat"
    os.environ["RECON_USE_GPU"] = "false"
    os.environ["RECON_MAX_IMAGE_SIZE"] = "1000"
    
    adapter = COLMAPAdapter()
    assert adapter._engine_path == "C:/colmap/COLMAP.bat"
    assert adapter._use_gpu is False
    assert adapter._max_image_size == 1000

@patch("modules.reconstruction_engine.adapter.subprocess.Popen")
@patch("modules.reconstruction_engine.adapter.shutil.copy2")
def test_colmap_adapter_run(mock_copy, mock_popen, tmp_path):
    engine_path = "colmap.exe"
    adapter = COLMAPAdapter(engine_path=engine_path)
    
    input_frames = []
    for name in ("frame1.jpg", "frame2.jpg", "frame3.jpg"):
        frame_path = tmp_path / name
        write_frame_and_mask(frame_path)
        input_frames.append(str(frame_path))
        
    output_dir = tmp_path / "workspace"
    output_dir.mkdir()

    dense_dir = output_dir / "dense"
    dense_dir.mkdir(parents=True)
    fused_ply = dense_dir / "fused.ply"
    fused_ply.write_bytes(b"ply\n" + (b"0" * 2048))
    mesh_file = dense_dir / "meshed-poisson.ply"
    mesh_file.write_text("ply\n", encoding="utf-8")

    mock_process = MagicMock()
    mock_process.returncode = 0
    # COLMAP 4.0.3 style output for the parser check
    mock_process.communicate.return_value = (
        "I20260413 ...] Registered images: 10\nI20260413 ...] Points: 1000", 
        ""
    )
    mock_popen.return_value = mock_process

    trimesh.creation.box().export(mesh_file)

    # Added match_mode_counts to the mocked prep result
    prep_mock = {
        "images_dir": tmp_path,
        "masks_dir": tmp_path,
        "accepted_frames": 10,
        "rejected_unreadable_frame": 0,
        "rejected_missing_mask": 0,
        "rejected_bad_mask": 0,
        "copied_mask_count": 10,
        "match_mode_counts": {"stem": 10, "legacy": 0, "none": 0}
    }

    with patch.object(adapter, "_mask_is_usable", return_value=True):
        with patch.object(adapter, "_frame_is_usable", return_value=True):
            # We need to mock _select_best_sparse_model because the test doesn't set up a real sparse dir
            with patch.object(adapter, "_select_best_sparse_model", return_value={"registered_images": 10, "points_3d": 1000, "path": tmp_path}):
                results = adapter.run_reconstruction(input_frames, output_dir)




    assert results["mesh_path"] == str(mesh_file)
    assert results["log_path"] == str(output_dir / "reconstruction.log")
    assert mock_copy.call_count == 6

    commands = [call.args[0][1] for call in mock_popen.call_args_list]
    assert commands == [
        "feature_extractor",
        "exhaustive_matcher",
        "mapper",
        "image_undistorter",
        "patch_match_stereo",
        "stereo_fusion",
        "poisson_mesher",
    ]

def test_colmap_artifact_discovery_requires_real_mesh(tmp_path):
    adapter = COLMAPAdapter(engine_path="fake")
    output_dir = tmp_path / "workspace"
    output_dir.mkdir()
    
    dense_dir = output_dir / "dense"
    dense_dir.mkdir(parents=True)
    fused_ply = dense_dir / "fused.ply"
    fused_ply.write_bytes(b"ply\n" + (b"1" * 2048))

    prep = {
        "images_dir": output_dir / "images",
        "masks_dir": output_dir / "masks",
        "accepted_frames": 10,
        "rejected_missing_mask": 0,
        "rejected_bad_mask": 0,
        "rejected_unreadable_frame": 0,
        "copied_mask_count": 10,
        "match_mode_counts": {"stem": 10, "legacy": 0, "none": 0}
    }
    prep["images_dir"].mkdir()
    prep["masks_dir"].mkdir()

    with patch.object(adapter, "_prepare_workspace", return_value=prep):
        with patch("modules.reconstruction_engine.adapter.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = iter([])
            mock_popen.return_value = mock_process

            with patch.object(adapter, "_select_best_sparse_model", return_value={"registered_images": 10, "points_3d": 1000, "path": tmp_path}):
                with pytest.raises(RuntimeError, match="no usable mesh artifacts"):
                    adapter.run_reconstruction([], output_dir)


def test_colmap_adapter_rejects_unreadable_frames(tmp_path):
    adapter = COLMAPAdapter(engine_path="fake")
    output_dir = tmp_path / "workspace"
    output_dir.mkdir()

    input_frames = []
    for name in ("frame1.jpg", "frame2.jpg", "frame3.jpg"):
        frame_path = tmp_path / name
        write_frame_and_mask(frame_path)
        input_frames.append(str(frame_path))

    with patch.object(adapter, "_frame_is_usable", return_value=False):
        with patch.object(adapter, "_mask_is_usable", return_value=True):
            with pytest.raises(InsufficientInputError, match="unreadable=3"):
                adapter.run_reconstruction(input_frames, output_dir)

def test_colmap_adapter_dense_masking_flow(tmp_path):
    """Verifies that the adapter correctly triggers dynamic masking when undistorted images exist."""
    import cv2
    import numpy as np
    from modules.reconstruction_engine.failures import DenseMaskAlignmentError
    
    adapter = COLMAPAdapter(engine_path="fake")
    output_dir = tmp_path / "workspace"
    output_dir.mkdir()
    
    dense_dir = output_dir / "dense"
    dense_images_dir = dense_dir / "images"
    dense_images_dir.mkdir(parents=True)
    
    # Create a dummy undistorted image
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_img[20:80, 20:80] = [255, 255, 255]
    img_path = dense_images_dir / "frame_001.jpg"
    _, buff = cv2.imencode(".jpg", dummy_img)
    buff.tofile(str(img_path))
    
    # Mock dependencies to reach the dense masking stage
    prep = {
        "images_dir": output_dir / "images",
        "masks_dir": output_dir / "masks", # effective_masks_dir must exist
        "accepted_frames": 10,
        "rejected_unreadable_frame": 0,
        "rejected_missing_mask": 0,
        "rejected_bad_mask": 0,
        "copied_mask_count": 10,
        "match_mode_counts": {"stem": 10, "legacy": 0, "none": 0}
    }
    prep["masks_dir"].mkdir()
    
    best_model = {"path": tmp_path, "registered_images": 10, "points_3d": 1000}
    
    with patch.object(adapter, "_prepare_workspace", return_value=prep):
        with patch.object(adapter, "_run_command"):
            with patch.object(adapter, "_select_best_sparse_model", return_value=best_model):
                with patch.object(adapter, "_validate_dense_workspace", return_value=1000):
                    with patch.object(adapter, "_discover_mesh_candidates", return_value=["mesh.ply"]):
                        adapter.run_reconstruction([], output_dir)
    
    # Verify mask was generated
    expected_mask = dense_dir / "stereo" / "masks" / "frame_001.jpg.png"
    assert expected_mask.exists()
    
    # Verify it's a valid mask
    mask_array = np.fromfile(str(expected_mask), np.uint8)
    mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (100, 100)
    assert np.any(mask > 0)

def test_colmap_adapter_high_fallback_triggers_unmasked_fusion(tmp_path):
    """Verifies that if >30% of frames fallback to white masks, stereo fusion is unmasked."""
    import cv2
    import numpy as np
    
    adapter = COLMAPAdapter(engine_path="fake")
    output_dir = tmp_path / "workspace"
    output_dir.mkdir()
    
    dense_dir = output_dir / "dense"
    dense_images_dir = dense_dir / "images"
    dense_images_dir.mkdir(parents=True)
    
    # Create 10 images: 4 empty (40% fallback ratio)
    for i in range(10):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        if i < 4:
            # Leave black -> occupancy 0 -> fallback
            pass
        else:
            # Subject -> occupancy > MIN -> normal
            img[20:80, 20:80] = [255, 255, 255]
            
        img_path = dense_images_dir / f"frame_{i:03d}.jpg"
        _, buff = cv2.imencode(".jpg", img)
        buff.tofile(str(img_path))
    
    prep = {
        "images_dir": output_dir / "images",
        "masks_dir": output_dir / "masks",
        "accepted_frames": 10,
        "rejected_unreadable_frame": 0,
        "rejected_missing_mask": 0,
        "rejected_bad_mask": 0,
        "copied_mask_count": 10,
        "match_mode_counts": {"stem": 10, "legacy": 0, "none": 0}
    }
    prep["masks_dir"].mkdir()
    
    best_model = {"path": tmp_path, "registered_images": 10, "points_3d": 1000}
    
    with patch.object(adapter, "_prepare_workspace", return_value=prep):
        with patch.object(adapter, "_run_command") as mock_run:
            with patch.object(adapter, "_select_best_sparse_model", return_value=best_model):
                with patch.object(adapter, "_validate_dense_workspace", return_value=1000):
                    with patch.object(adapter, "_discover_mesh_candidates", return_value=["mesh.ply"]):
                        # Spy on stereo_fusion call
                        with patch.object(adapter.builder, "stereo_fusion", wraps=adapter.builder.stereo_fusion) as mock_fuse:
                            adapter.run_reconstruction([], output_dir)
                            
                            # check if mask_path was None due to force_unmasked_fusion
                            # find stereo_fusion call in mock_fuse
                            # check kwargs
                            _, kwargs = mock_fuse.call_args
                            assert kwargs["mask_path"] is None, "Should have reverted to unmasked fusion for >30% fallback"
