import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.reconstruction_engine.runner import ReconstructionRunner
from modules.reconstruction_engine.adapter import ReconstructionAdapter
from modules.shared_contracts.models import ReconstructionJob, ReconstructionAttemptType
from modules.reconstruction_engine.failures import InsufficientReconstructionError
from modules.operations.settings import settings

class MockAdapter(ReconstructionAdapter):
    @property
    def engine_type(self): return "mock"
    @property
    def is_stub(self): return False

    def __init__(self):
        self.attempts = 0
        self.last_density = None
        self.last_enforce_masks = None

    def run_reconstruction(self, input_frames, output_dir, density=1.0, enforce_masks=True):
        self.attempts += 1
        self.last_density = density
        self.last_enforce_masks = enforce_masks
        
        # Simulate failure for default (density=0.5)
        if density == 0.5:
            raise InsufficientReconstructionError("Default was too weak")
        
        # Success for denser (density=1.0)
        mesh_path = output_dir / "mesh.ply"
        mesh_path.write_text("v 0 0 0\nf 1 1 1")
        log_path = output_dir / "recon.log"
        log_path.write_text("success")
        
        return {
            "mesh_path": str(mesh_path),
            "texture_path": str(output_dir / "texture.png"),
            "log_path": str(log_path),
            "registered_images": 50,
            "sparse_points": 1000,
            "dense_points_fused": 5000,
            "mesher_used": "poisson"
        }

@pytest.fixture
def session_dir(tmp_path):
    d = tmp_path / "sessions" / "cap_test"
    d.mkdir(parents=True)
    (d / "frames").mkdir()
    for i in range(10):
        (d / "frames" / f"frame_{i:04d}.jpg").write_text("fake image data")
    return d

def test_fallback_to_denser_frames(session_dir, tmp_path):
    # Setup settings for fallback
    with patch.object(settings, "recon_fallback_steps", ["default", "denser_frames"]), \
         patch.object(settings, "recon_fallback_sample_rate", 5):
        
        adapter = MockAdapter()
        runner = ReconstructionRunner(adapter=adapter)
        
        video_path = session_dir / "video" / "raw_video.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_text("fake video data")

        initial_frames = [str(f) for f in (session_dir / "frames").glob("*.jpg")]
        job = ReconstructionJob(
            job_id="job_test",
            capture_session_id="cap_test",
            product_id="prod_test",
            input_frames=initial_frames,
            source_video_path=str(video_path),
            job_dir=str(tmp_path / "jobs" / "job_test")
        )
        
        # We want to prove denser fallback uses MORE frames.
        # So we mock FrameExtractor to return a larger list.
        denser_frames = [f"frame_{i:04d}.jpg" for i in range(50)]
        
        # Mocking _validate_input_frames and _validate_mesh_artifact to avoid real trimesh/CV2 calls
        with patch.object(ReconstructionRunner, "_validate_input_frames", return_value=job.input_frames), \
             patch.object(ReconstructionRunner, "_validate_mesh_artifact", return_value=(100, 200)), \
             patch("modules.reconstruction_engine.runner.calculate_checksum", return_value="fake_hash"), \
             patch("modules.capture_workflow.frame_extractor.FrameExtractor") as MockExtractor:
            
            # Setup the mock extractor instance
            mock_inst = MockExtractor.return_value
            mock_inst.extract_keyframes.return_value = denser_frames
            mock_inst.thresholds = MagicMock()
            
            manifest = runner.run(job)
            
            assert adapter.attempts == 2
            
            # Verify MockExtractor was called for the second attempt
            assert MockExtractor.call_count == 1
            assert mock_inst.thresholds.frame_sample_rate == 5
            
            # Check audit trail
            audit_path = Path(job.job_dir) / "reconstruction_audit.json"
            assert audit_path.exists()
            import json
            with open(audit_path, "r") as f:
                audit_data = json.load(f)
                assert len(audit_data["attempts"]) == 2
                
                # Default attempt
                assert audit_data["attempts"][0]["attempt_type"] == "default"
                
                # Denser attempt
                assert audit_data["attempts"][1]["attempt_type"] == "denser_frames"
                assert audit_data["attempts"][1]["frames_used"] == 50 
                assert audit_data["attempts"][1]["status"] == "success"
                assert audit_data["attempts"][1]["sampling_rate_used"] == 5
                # Compare only relevant parts to avoid Windows encoding issues with 'Ömer'
                actual_video_path = Path(audit_data["attempts"][1]["source_video_path"])
                assert actual_video_path.name == video_path.name
                assert actual_video_path.parent.name == "video"
                assert "extracted_frames" in audit_data["attempts"][1]["reextracted_frames_dir"]
                
                assert audit_data["selected_best_index"] == 1

def test_unmasked_fallback_opt_in(session_dir, tmp_path):
    # 1. Unmasked disabled
    with patch.object(settings, "recon_fallback_steps", ["default", "unmasked"]), \
         patch.object(settings, "recon_unmasked_fallback_enabled", False):
        
        adapter = MockAdapter()
        runner = ReconstructionRunner(adapter=adapter)
        
        job = ReconstructionJob(
            job_id="job_unmasked",
            capture_session_id="cap_test",
            product_id="prod_test",
            input_frames=[str(f) for f in (session_dir / "frames").glob("*.jpg")],
            job_dir=str(tmp_path / "jobs" / "job_unmasked")
        )
        
        # Setup MockAdapter to fail on first call if density is not what we want
        adapter.run_reconstruction = MagicMock(side_effect=InsufficientReconstructionError("weak"))
        
        with patch.object(ReconstructionRunner, "_validate_input_frames", return_value=job.input_frames), \
             pytest.raises(InsufficientReconstructionError):
            runner.run(job)
            
        assert adapter.run_reconstruction.call_count == 1 # Only default tried, unmasked skipped

    # 2. Unmasked enabled
    with patch.object(settings, "recon_fallback_steps", ["default", "unmasked"]), \
         patch.object(settings, "recon_unmasked_fallback_enabled", True):
        
        adapter = MockAdapter()
        # Mock run_reconstruction to check enforce_masks
        adapter.run_reconstruction = MagicMock(side_effect=[
            InsufficientReconstructionError("weak"),
            {
                "mesh_path": "fake", "texture_path": "fake", "log_path": "fake",
                "registered_images": 10, "sparse_points": 10, "dense_points_fused": 10, "mesher_used": "poisson"
            }
        ])
        
        runner = ReconstructionRunner(adapter=adapter)
        
        with patch.object(ReconstructionRunner, "_validate_input_frames", return_value=job.input_frames), \
             patch.object(ReconstructionRunner, "_validate_mesh_artifact", return_value=(10, 20)), \
             patch("modules.reconstruction_engine.runner.calculate_checksum", return_value="h"):
            runner.run(job)
            
        assert adapter.run_reconstruction.call_count == 2
        # Check call args of second call (index 1)
        # args = (frames, dir, density, enforce_masks)
        _, kwargs = adapter.run_reconstruction.call_args_list[1]
        assert kwargs["enforce_masks"] is False

def test_honest_failure_no_silent_success(session_dir, tmp_path):
    # All attempts fail
    with patch.object(settings, "recon_fallback_steps", ["default", "denser_frames"]):
        adapter = MockAdapter()
        adapter.run_reconstruction = MagicMock(side_effect=InsufficientReconstructionError("total failure"))
        
        runner = ReconstructionRunner(adapter=adapter)
        job = ReconstructionJob(
            job_id="job_fail",
            capture_session_id="cap_test",
            product_id="prod_test",
            input_frames=[str(f) for f in (session_dir / "frames").glob("*.jpg")],
            job_dir=str(tmp_path / "jobs" / "job_fail")
        )
        
        with patch.object(ReconstructionRunner, "_validate_input_frames", return_value=job.input_frames), \
             pytest.raises(InsufficientReconstructionError) as excinfo:
            runner.run(job)
            
        assert "All fallback attempts failed" in str(excinfo.value)
        
        # Audit should exist and mark final_status as recapture_required
        audit_path = Path(job.job_dir) / "reconstruction_audit.json"
        import json
        with open(audit_path, "r") as f:
            audit_data = json.load(f)
            assert audit_data["final_status"] == "recapture_required"
