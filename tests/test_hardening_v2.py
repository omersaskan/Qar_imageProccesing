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
from PIL import Image

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

def test_reconstruction_triggers_extraction_if_missing(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "captures").mkdir()
    (data_root / "reconstructions").mkdir()
    
    worker = IngestionWorker(data_root=str(data_root))
    
    session_id = "test_trigger_extraction"
    session_dir = data_root / "captures" / session_id
    session_dir.mkdir()
    (session_dir / "video").mkdir()
    video_path = session_dir / "video" / "raw_video.mp4"
    video_path.write_text("dummy video")
    
    # Session starts CAPTURED but has no frames
    session = CaptureSession(
        session_id=session_id,
        product_id="p1",
        operator_id="o1",
        status=AssetStatus.CAPTURED,
        extracted_frames=[] # EMPTY
    )
    
    with pytest.MonkeyPatch().context() as m:
        # 1. Mock _handle_frame_extraction
        def mock_extract(s):
            frames_dir = data_root / "captures" / s.session_id / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            f1 = frames_dir / "f1.jpg"
            f1.write_text("data")
            s.extracted_frames = [str(f1)]
            return s
        m.setattr(worker, "_handle_frame_extraction", mock_extract)
        
        # 2. Mock persistence and other side effects
        m.setattr(worker, "_persist_session", lambda s, **f: s)
        m.setattr(worker, "_update_guidance", lambda s: None)
        
        # 3. Mock JobManager to just capture the draft
        captured_draft = None
        class MockManager:
            def __init__(self, **kwargs): pass
            def create_job(self, draft):
                nonlocal captured_draft
                captured_draft = draft
                # Mock a job object
                class MockJob:
                    job_id = draft.job_id
                    job_dir = str(data_root / "reconstructions" / draft.job_id)
                return MockJob()
            def update_job_status(self, *args, **kwargs): pass
            
        m.setattr("modules.reconstruction_engine.job_manager.JobManager", MockManager)
        
        # 4. Mock Runner
        class MockRunner:
            def run(self, job):
                return OutputManifest(
                    job_id=job.job_id, mesh_path="m.ply", log_path="l.txt", processing_time_seconds=1.0
                )
        m.setattr("modules.reconstruction_engine.runner.ReconstructionRunner", MockRunner)
        
        # 5. Mock coverage
        (session_dir / "reports").mkdir()
        with open(session_dir / "reports" / "coverage_report.json", "w") as f:
            json.dump({"overall_status": "sufficient", "coverage_score": 0.9}, f)
            
        # 6. EXECUTE
        worker._handle_reconstruction(session)
        
        assert captured_draft is not None
        assert len(captured_draft.input_frames) == 1
        assert "f1.jpg" in captured_draft.input_frames[0]

def test_budget_retry_uses_reconstruction_job_id(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "captures").mkdir()
    (data_root / "reconstructions").mkdir()
    
    worker = IngestionWorker(data_root=str(data_root))
    
    session_id = "test_budget_job_id"
    # Create a job with a unique ID
    unique_job_id = f"job_{session_id}_999999"
    manager = JobManager(data_root=str(data_root))
    
    # We need a draft to create a job
    draft = ReconstructionJobDraft(
        job_id=unique_job_id,
        capture_session_id=session_id,
        input_frames=["f1.jpg"],
        product_id="p1"
    )
    manager.create_job(draft)
    
    session = CaptureSession(
        session_id=session_id,
        product_id="p1",
        operator_id="o1",
        status=AssetStatus.PROCESSING_BUDGET_EXCEEDED,
        reconstruction_job_id=unique_job_id
    )
    
    with pytest.MonkeyPatch().context() as m:
        # Mock remesh_retry
        m.setattr("modules.reconstruction_engine.runner.ReconstructionRunner.remesh_retry", 
                  lambda self, job, depth, trim: OutputManifest(job_id=job.job_id, mesh_path="m.ply", log_path="l.txt", processing_time_seconds=1.0))
        
        worker._handle_budget_exceeded_retry(session)
        # If it didn't raise IrrecoverableError, it found the job.

def test_unique_job_ids_no_collision(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "captures").mkdir()
    (data_root / "reconstructions").mkdir()
    
    worker = IngestionWorker(data_root=str(data_root))
    session_id = "test_collision"
    
    # Setup session and frames
    session_dir = data_root / "captures" / session_id
    session_dir.mkdir()
    frames_dir = session_dir / "frames"
    frames_dir.mkdir()
    Image.new("RGB", (1, 1)).save(frames_dir / "f1.jpg", "JPEG")
    
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
        
    job_ids = []
    with pytest.MonkeyPatch().context() as m:
        original_create = JobManager.create_job
        def wrapped_create(self, draft):
            job_ids.append(draft.job_id)
            return original_create(self, draft)
        m.setattr(JobManager, "create_job", wrapped_create)
        m.setattr("modules.reconstruction_engine.runner.ReconstructionRunner.run", lambda self, job: OutputManifest(
            job_id=job.job_id, mesh_path="m.ply", log_path="l.txt", processing_time_seconds=1.0
        ))
        
        # Trigger two attempts immediately, clearing job_id between to force new job generation
        worker._handle_reconstruction(session)
        session.reconstruction_job_id = None
        worker._handle_reconstruction(session)
        
        assert len(job_ids) == 2
        assert job_ids[0] != job_ids[1]
        assert "job_" in job_ids[0]

def test_coverage_recapture_reasons(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "captures").mkdir()
    
    worker = IngestionWorker(data_root=str(data_root))
    session_id = "test_coverage_reasons"
    session_dir = data_root / "captures" / session_id
    session_dir.mkdir()
    frames_dir = session_dir / "frames"
    frames_dir.mkdir()
    f1 = frames_dir / "f1.jpg"
    Image.new("RGB", (1, 1)).save(f1, "JPEG")
    
    session = CaptureSession(
        session_id=session_id,
        product_id="p1",
        operator_id="o1",
        status=AssetStatus.CAPTURED,
        extracted_frames=[str(f1)]
    )
    
    # Provide coverage report with hard_reasons
    (session_dir / "reports").mkdir()
    with open(session_dir / "reports" / "coverage_report.json", "w") as f:
        json.dump({
            "overall_status": "insufficient", 
            "hard_reasons": ["Missing top view", "Too much blur"],
            "coverage_score": 0.4
        }, f)
        
    updated = worker._handle_reconstruction(session)
    
    assert updated.status == AssetStatus.RECAPTURE_REQUIRED
    assert "Missing top view" in updated.failure_reason
    assert "Too much blur" in updated.failure_reason
    assert updated.coverage_score == 0.4

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
    for f_name in ["f1.jpg", "f2.jpg", "f3.jpg"]:
        Image.new("RGB", (1, 1)).save(frames_dir / f_name, "JPEG")

    
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
        session.reconstruction_job_id = None
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
    for f_name in ["f1.jpg", "f2.jpg", "f3.jpg"]:
        Image.new("RGB", (1, 1)).save(frames_dir / f_name, "JPEG")

    
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
