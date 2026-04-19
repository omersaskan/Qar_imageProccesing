import os
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from modules.reconstruction_engine.failures import InsufficientInputError, MissingArtifactError
from modules.reconstruction_engine.job_manager import JobManager
from modules.reconstruction_engine.runner import ReconstructionRunner
from modules.shared_contracts.models import ReconstructionJobDraft


def _write_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (100, 100), (255, 200, 100))
    img.save(path)
    
    masks_dir = path.parent / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_path = masks_dir / f"{path.stem}.png"
    # Create a mask with ~25% occupancy (50x50 white square in 100x100)
    mask = Image.new("L", (100, 100), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([25, 25, 75, 75], fill=255)
    mask.save(mask_path)


class MeshWritingAdapter:
    engine_type = "colmap"
    is_stub = False

    def __init__(self):
        self.last_output_dir = None

    def run_reconstruction(self, input_frames, output_dir: Path, **kwargs) -> dict:
        self.last_output_dir = Path(output_dir)
        mesh_path = output_dir / "raw_mesh.obj"
        log_path = output_dir / "reconstruction.log"
        mesh_path.write_text(
            "v 0 0 0\n"
            "v 1 0 0\n"
            "v 0 1 0\n"
            "f 1 2 3\n",
            encoding="utf-8",
        )
        log_path.write_text("ok\n", encoding="utf-8")
        return {
            "mesh_path": str(mesh_path),
            "texture_path": str(output_dir / "_no_texture.png"),
            "log_path": str(log_path),
        }


class PointCloudLikeAdapter:
    engine_type = "colmap"
    is_stub = False

    def run_reconstruction(self, input_frames, output_dir: Path, **kwargs) -> dict:
        mesh_path = output_dir / "raw_mesh.obj"
        log_path = output_dir / "reconstruction.log"
        mesh_path.write_text(
            "v 0 0 0\n"
            "v 1 0 0\n"
            "v 0 1 0\n",
            encoding="utf-8",
        )
        log_path.write_text("ok\n", encoding="utf-8")
        return {
            "mesh_path": str(mesh_path),
            "texture_path": str(output_dir / "_no_texture.png"),
            "log_path": str(log_path),
        }


def test_runner_production_guard(monkeypatch):
    from modules.operations.settings import settings, AppEnvironment
    monkeypatch.setattr(settings, "env", AppEnvironment.PRODUCTION)
    monkeypatch.setattr(settings, "recon_pipeline", "simulated")
    monkeypatch.setattr(settings, "pilot_api_key", "sk_test_mock")

    with patch("pathlib.Path.exists", return_value=True):
        with patch("os.path.exists", return_value=True):
            with pytest.raises(RuntimeError, match="strictly prohibited"):
                # Accessing .adapter property to trigger the guard
                _ = ReconstructionRunner().adapter


def test_runner_production_missing_path(monkeypatch):
    from modules.operations.settings import settings, AppEnvironment
    monkeypatch.setattr(settings, "env", AppEnvironment.PRODUCTION)
    monkeypatch.setattr(settings, "recon_pipeline", "colmap_dense")
    monkeypatch.setattr(settings, "pilot_api_key", "sk_test_mock")
    monkeypatch.delenv("RECON_ENGINE_PATH", raising=False)

    with patch("pathlib.Path.exists", return_value=False):
        with patch("os.path.exists", return_value=False):
            with pytest.raises(RuntimeError, match="must be configured"):
                # Accessing .adapter property to trigger the validation
                _ = ReconstructionRunner().adapter


def test_runner_rejects_simulated_without_explicit_opt_in(monkeypatch):
    from modules.operations.settings import settings, AppEnvironment
    monkeypatch.setattr(settings, "env", AppEnvironment.LOCAL_DEV)
    monkeypatch.setattr(settings, "recon_pipeline", "simulated")
    monkeypatch.delenv("ALLOW_SIMULATED_RECONSTRUCTION", raising=False)

    with pytest.raises(RuntimeError, match="disabled by default"):
        # Trigger guard in property
        _ = ReconstructionRunner().adapter


def test_runner_success(tmp_path, monkeypatch):
    from modules.operations.settings import settings
    monkeypatch.setattr(settings, "env", "development")
    monkeypatch.setattr(settings, "recon_pipeline", "simulated")
    monkeypatch.setattr(settings, "recon_fallback_steps", ["default"])
    monkeypatch.setenv("ALLOW_SIMULATED_RECONSTRUCTION", "true")

    input_frames = []
    for index in range(3):
        frame_path = tmp_path / f"frame_{index}.jpg"
        _write_frame(frame_path)
        input_frames.append(str(frame_path))

    manager = JobManager(data_root=str(tmp_path))
    draft = ReconstructionJobDraft(
        job_id="RJ_003",
        capture_session_id="S1",
        input_frames=input_frames,
        product_id="P1",
    )
    job = manager.create_job(draft)

    runner = ReconstructionRunner()
    manifest = runner.run(job)

    assert manifest.job_id == "RJ_003"
    assert "raw_mesh.obj" in manifest.mesh_path
    assert manifest.mesh_metadata.vertex_count == 3
    assert manifest.mesh_metadata.face_count == 1
    assert manifest.checksum
    assert (Path(job.job_dir) / "manifest.json").exists()


def test_runner_insufficient_input(tmp_path):
    manager = JobManager(data_root=str(tmp_path))
    draft = ReconstructionJobDraft(
        job_id="RJ_004",
        capture_session_id="S1",
        input_frames=["f1.jpg"],
        product_id="P1",
    )
    job = manager.create_job(draft)

    runner = ReconstructionRunner(adapter=MeshWritingAdapter())
    with pytest.raises(InsufficientInputError):
        runner.run(job)


def test_runner_rejects_non_mesh_output(tmp_path):
    input_frames = []
    for index in range(3):
        frame_path = tmp_path / f"frame_{index}.jpg"
        _write_frame(frame_path)
        input_frames.append(str(frame_path))

    manager = JobManager(data_root=str(tmp_path))
    draft = ReconstructionJobDraft(
        job_id="RJ_005",
        capture_session_id="S1",
        input_frames=input_frames,
        product_id="P1",
    )
    job = manager.create_job(draft)

    runner = ReconstructionRunner(adapter=PointCloudLikeAdapter())
    with pytest.raises(MissingArtifactError, match="not a polygon mesh"):
        runner.run(job)


def test_runner_mirrors_non_ascii_workspace_for_external_tools(tmp_path):
    if os.name != "nt":
        pytest.skip("ASCII workspace mirroring is only relevant on Windows paths.")

    input_frames = []
    for index in range(3):
        frame_path = tmp_path / f"frame_{index}.jpg"
        _write_frame(frame_path)
        input_frames.append(str(frame_path))

    non_ascii_root = tmp_path / "\u00d6zelKlasor"
    manager = JobManager(data_root=str(non_ascii_root))
    draft = ReconstructionJobDraft(
        job_id="RJ_006",
        capture_session_id="S1",
        input_frames=input_frames,
        product_id="P1",
    )
    job = manager.create_job(draft)

    adapter = MeshWritingAdapter()
    runner = ReconstructionRunner(adapter=adapter)
    manifest = runner.run(job)

    assert adapter.last_output_dir is not None
    assert all(ord(ch) < 128 for ch in str(adapter.last_output_dir))
    assert manifest.mesh_path.startswith(str(Path(job.job_dir).resolve()))
    assert Path(manifest.mesh_path).exists()
