import pytest
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from modules.operations.worker import IngestionWorker
from modules.shared_contracts.models import CaptureSession, ReconstructionJobDraft
from modules.shared_contracts.lifecycle import AssetStatus, ReconstructionStatus
from modules.reconstruction_engine.output_manifest import OutputManifest
from modules.reconstruction_engine.job_manager import JobManager

def test_output_manifest_safety_defaults():
    # Verify that the new safety fields have the correct defaults
    manifest = OutputManifest(
        job_id="test_job",
        mesh_path="mesh.ply",
        log_path="log.txt",
        processing_time_seconds=10.5
    )
    assert manifest.ai_generated is False
    assert manifest.geometry_source == "photogrammetry"
    assert manifest.production_status == "production_candidate"
    assert manifest.requires_manual_review is False
    assert manifest.may_override_recapture_required is False

def test_handle_reconstruction_prioritizes_extracted_frames(tmp_path):
    # Setup data root
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "captures").mkdir()
    (data_root / "reconstructions").mkdir()
    
    worker = IngestionWorker(data_root=str(data_root))
    
    session_id = "test_session"
    session_dir = data_root / "captures" / session_id
    session_dir.mkdir()
    frames_dir = session_dir / "frames"
    frames_dir.mkdir()
    
    # Create some dummy frames
    frame1 = frames_dir / "frame1.jpg"
    frame1.write_text("dummy")
    frame2 = frames_dir / "frame2.jpg"
    frame2.write_text("dummy")
    
    # Session with explicit frames
    session = CaptureSession(
        session_id=session_id,
        product_id="prod1",
        operator_id="op1",
        status=AssetStatus.CAPTURED,
        extracted_frames=[str(frame1)]
    )
    
    # Mocking JobManager.create_job to capture what was passed
    original_create = JobManager.create_job
    captured_draft = None
    
    def mock_create_job(self, draft):
        nonlocal captured_draft
        captured_draft = draft
        # Return a dummy job to prevent further execution
        from modules.shared_contracts.models import ReconstructionJob
        return ReconstructionJob(
            job_id=draft.job_id,
            capture_session_id=draft.capture_session_id,
            product_id=draft.product_id,
            input_frames=draft.input_frames,
            job_dir=str(Path(self.reconstructions_dir) / draft.job_id)
        )
    
    # We also need to mock runner.run and other things to avoid real execution
    with pytest.MonkeyPatch().context() as m:
        # Instead of mocking create_job entirely, we just capture the draft
        original_create = JobManager.create_job
        def wrapped_create(self, draft):
            nonlocal captured_draft
            captured_draft = draft
            return original_create(self, draft)
            
        m.setattr(JobManager, "create_job", wrapped_create)
        m.setattr("modules.reconstruction_engine.runner.ReconstructionRunner.run", lambda self, job: OutputManifest(
            job_id=job.job_id, mesh_path="m.ply", log_path="l.txt", processing_time_seconds=1.0
        ))
        
        # We need a coverage report
        (session_dir / "reports").mkdir()
        with open(session_dir / "reports" / "coverage_report.json", "w") as f:
            json.dump({"overall_status": "sufficient", "coverage_score": 0.9}, f)
            
        worker._handle_reconstruction(session)
        
        assert captured_draft is not None
        assert captured_draft.input_frames == [str(frame1)]
        assert "frame2.jpg" not in captured_draft.input_frames

def test_handle_reconstruction_unique_job_id(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "captures").mkdir()
    (data_root / "reconstructions").mkdir()
    
    worker = IngestionWorker(data_root=str(data_root))
    
    session_id = "test_session_unique"
    session_dir = data_root / "captures" / session_id
    session_dir.mkdir()
    frames_dir = session_dir / "frames"
    frames_dir.mkdir()
    (frames_dir / "f1.jpg").write_text("d")
    (frames_dir / "f2.jpg").write_text("d")
    (frames_dir / "f3.jpg").write_text("d")
    
    session = CaptureSession(
        session_id=session_id,
        product_id="p1",
        operator_id="o1",
        status=AssetStatus.CAPTURED,
        extracted_frames=[str(frames_dir / "f1.jpg")]
    )
    
    (session_dir / "reports").mkdir()
    with open(session_dir / "reports" / "coverage_report.json", "w") as f:
        json.dump({"overall_status": "sufficient", "coverage_score": 0.9}, f)
        
    captured_ids = []
    original_create = JobManager.create_job
    def mock_create(self, draft):
        captured_ids.append(draft.job_id)
        from modules.shared_contracts.models import ReconstructionJob
        return ReconstructionJob(
            job_id=draft.job_id,
            capture_session_id=draft.capture_session_id,
            product_id=draft.product_id,
            input_frames=draft.input_frames,
            job_dir=str(Path(self.reconstructions_dir) / draft.job_id)
        )
        
    with pytest.MonkeyPatch().context() as m:
        original_create = JobManager.create_job
        def wrapped_create(self, draft):
            captured_ids.append(draft.job_id)
            return original_create(self, draft)
            
        m.setattr(JobManager, "create_job", wrapped_create)
        m.setattr("modules.reconstruction_engine.runner.ReconstructionRunner.run", lambda self, job: OutputManifest(
            job_id=job.job_id, mesh_path="m.ply", log_path="l.txt", processing_time_seconds=1.0
        ))
        
        worker._handle_reconstruction(session)
        time.sleep(1.1)
        worker._handle_reconstruction(session)
        
        assert len(captured_ids) == 2
        assert captured_ids[0] != captured_ids[1]
        assert captured_ids[0].startswith("job_test_session_unique_")

def test_runner_failure_marks_job_failed(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "captures").mkdir()
    (data_root / "reconstructions").mkdir()
    
    worker = IngestionWorker(data_root=str(data_root))
    
    session_id = "test_session_fail"
    session_dir = data_root / "captures" / session_id
    session_dir.mkdir()
    frames_dir = session_dir / "frames"
    frames_dir.mkdir()
    (frames_dir / "f1.jpg").write_text("d")
    (frames_dir / "f2.jpg").write_text("d")
    (frames_dir / "f3.jpg").write_text("d")
    
    session = CaptureSession(
        session_id=session_id,
        product_id="p1",
        operator_id="o1",
        status=AssetStatus.CAPTURED,
        extracted_frames=[str(frames_dir / "f1.jpg"), str(frames_dir / "f2.jpg"), str(frames_dir / "f3.jpg")]
    )
    
    (session_dir / "reports").mkdir()
    with open(session_dir / "reports" / "coverage_report.json", "w") as f:
        json.dump({"overall_status": "sufficient", "coverage_score": 0.9}, f)
        
    def mock_run_fail(self, job):
        raise RuntimeError("RECON_CRASH")
        
    with pytest.MonkeyPatch().context() as m:
        m.setattr("modules.reconstruction_engine.runner.ReconstructionRunner.run", mock_run_fail)
        
        try:
            worker._handle_reconstruction(session)
        except Exception:
            pass
            
        # Check job status
        job_manager = JobManager(data_root=str(data_root))
        # We need the actual job_id used, which has a timestamp. 
        # Since we only have one job, let's find it.
        recon_dirs = list((data_root / "reconstructions").iterdir())
        assert len(recon_dirs) == 1
        job_id = recon_dirs[0].name
        job = job_manager.get_job(job_id)
        assert job.status == ReconstructionStatus.FAILED
        assert "RECON_CRASH" in job.failure_reason
