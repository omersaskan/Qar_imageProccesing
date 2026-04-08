from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, HttpUrl
from modules.shared_contracts.models import (
    CaptureSession, 
    ReconstructionJob, 
    ReconstructionJobDraft
)
from modules.shared_contracts.lifecycle import AssetStatus

class Packager:
    def __init__(self):
        pass

    def prepare_reconstruction_payload(self, 
                                     session: CaptureSession, 
                                     extracted_frames: List[str],
                                     quality_report: Optional[Dict[str, Any]] = None,
                                     coverage_report: Optional[Dict[str, Any]] = None) -> ReconstructionJobDraft:
        """
        Packaging the finalized capture session for reconstruction.
        """
        # Ensure job ID is unique (e.g., RJ-session_id)
        job_id = f"RJ-{session.session_id}"
        
        return ReconstructionJobDraft(
            job_id=job_id,
            capture_session_id=session.session_id,
            input_frames=extracted_frames,
            quality_report=quality_report or {},
            coverage_report=coverage_report or {},
            product_id=session.product_id
        )
