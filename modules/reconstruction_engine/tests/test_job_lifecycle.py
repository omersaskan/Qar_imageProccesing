import pytest
import json
from pathlib import Path
from shared_contracts.models import ReconstructionJobDraft
from shared_contracts.lifecycle import ReconstructionStatus
from reconstruction_engine.job_manager import JobManager

def test_create_and_load_job(tmp_path):
    manager = JobManager(data_root=str(tmp_path))
    draft = ReconstructionJobDraft(
        job_id="RJ_001",
        capture_session_id="S1",
        input_frames=["f1.jpg", "f2.jpg"],
        product_id="P1"
    )
    
    job = manager.create_job(draft)
    assert job.status == ReconstructionStatus.QUEUED
    assert (Path(job.job_dir) / "job.json").exists()
    
    # Reload
    loaded_job = manager.get_job("RJ_001")
    assert loaded_job.job_id == "RJ_001"
    assert loaded_job.product_id == "P1"

def test_job_status_transitions(tmp_path):
    manager = JobManager(data_root=str(tmp_path))
    draft = ReconstructionJobDraft(
        job_id="RJ_002",
        capture_session_id="S1",
        input_frames=["f1.jpg", "f2.jpg"],
        product_id="P1"
    )
    manager.create_job(draft)
    
    # Running
    job = manager.update_job_status("RJ_002", ReconstructionStatus.RUNNING)
    assert job.status == ReconstructionStatus.RUNNING
    assert job.started_at is not None
    
    # Completed
    job = manager.update_job_status("RJ_002", ReconstructionStatus.COMPLETED)
    assert job.status == ReconstructionStatus.COMPLETED
    assert job.completed_at is not None
