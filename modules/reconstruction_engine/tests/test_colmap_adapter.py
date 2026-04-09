import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import trimesh
from modules.reconstruction_engine.adapter import COLMAPAdapter


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
    mock_process.stdout = iter(["loading...", "done."])
    mock_popen.return_value = mock_process

    trimesh.creation.box().export(mesh_file)

    with patch.object(adapter, "_mask_is_usable", return_value=True):
        with patch.object(adapter, "_frame_is_usable", return_value=True):
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
        "accepted_frames": 3,
        "rejected_missing_mask": 0,
        "rejected_bad_mask": 0,
        "rejected_unreadable_frame": 0,
    }
    prep["images_dir"].mkdir()
    prep["masks_dir"].mkdir()

    with patch.object(adapter, "_prepare_workspace", return_value=prep):
        with patch("modules.reconstruction_engine.adapter.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = iter([])
            mock_popen.return_value = mock_process

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
            with pytest.raises(RuntimeError, match="unreadable=3"):
                adapter.run_reconstruction(input_frames, output_dir)
