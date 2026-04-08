import os
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timezone
from modules.shared_contracts.models import ReconstructionJob, ReconstructionJobDraft
from modules.shared_contracts.lifecycle import ReconstructionStatus
from modules.utils.path_safety import validate_safe_path, ensure_dir, validate_identifier
from modules.utils.file_persistence import atomic_write_json, FileLock
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("job_manager")

class JobManager:
    def __init__(self, data_root: str = "data"):
        self.data_root = os.path.abspath(data_root)
        self.reconstructions_dir = os.path.join(self.data_root, "reconstructions")
        ensure_dir(self.reconstructions_dir)

    def create_job(self, draft: ReconstructionJobDraft) -> ReconstructionJob:
        """
        Creates a new ReconstructionJob from a draft and initializes the folder structure.
        """
        # 0. Input Validation
        if not draft.input_frames:
            raise ValueError(f"Job creation failed: No input frames provided for product {draft.product_id}")

        job_id = validate_identifier(draft.job_id, "Job ID")
        product_id = validate_identifier(draft.product_id, "Product ID")
        
        # 1. Path Safety
        job_dir = validate_safe_path(self.reconstructions_dir, job_id)
        
        # 2. Ensure unified storage directory exists
        ensure_dir(job_dir)
        ensure_dir(os.path.join(job_dir, "logs"))
        ensure_dir(os.path.join(job_dir, "temp"))

        # 2. Build the job model
        job = ReconstructionJob(
            job_id=job_id,
            capture_session_id=draft.capture_session_id,
            product_id=product_id,
            status=ReconstructionStatus.QUEUED,
            input_frames=draft.input_frames,
            job_dir=str(job_dir),
            created_at=datetime.now(timezone.utc)
        )

        # 3. Save job.json
        self._save_job_internal(job)
        return job

    def save_job(self, job: ReconstructionJob) -> None:
        """Saves job.json inside the job's directory atomically with a lock."""
        file_path = os.path.join(job.job_dir, "job.json")
        with FileLock(file_path):
            self._save_job_internal(job)

    def _save_job_internal(self, job: ReconstructionJob) -> None:
        """Internal helper to save job.json without acquiring a lock (assumes lock is held or not needed)."""
        file_path = os.path.join(job.job_dir, "job.json")
        atomic_write_json(file_path, job.model_dump(mode="json"))

    def get_job(self, job_id: str) -> Optional[ReconstructionJob]:
        """Loads job.json from the job's directory with validation."""
        job_id = validate_identifier(job_id, "Job ID")
        file_path = os.path.join(self.reconstructions_dir, job_id, "job.json")
        
        if not os.path.exists(file_path):
            logger.error(f"Job file not found: {file_path}")
            return None
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return ReconstructionJob.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to load job {job_id}: {e}")
            return None

    def update_job_status(self, job_id: str, status: ReconstructionStatus, failure_reason: Optional[str] = None) -> ReconstructionJob:
        job_id = validate_identifier(job_id, "Job ID")
        file_path = os.path.join(self.reconstructions_dir, job_id, "job.json")
        
        with FileLock(file_path):
            job = self.get_job(job_id)
            if not job:
                # User requirement: report failure clearly, don't just crash
                err_msg = f"Reconstruction job {job_id} not found in storage. Cannot update status to {status}."
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)

            job.status = status
            if status == ReconstructionStatus.RUNNING:
                job.started_at = datetime.now(timezone.utc)
            elif status in [ReconstructionStatus.COMPLETED, ReconstructionStatus.FAILED]:
                job.completed_at = datetime.now(timezone.utc)
                if failure_reason:
                    job.failure_reason = failure_reason

            self._save_job_internal(job)
            return job
