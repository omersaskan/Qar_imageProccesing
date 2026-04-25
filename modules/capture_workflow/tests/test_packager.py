import pytest
from datetime import datetime
from modules.shared_contracts.models import CaptureSession, ReconstructionJobDraft
from modules.capture_workflow.packager import Packager
from modules.shared_contracts.lifecycle import AssetStatus

def test_packager_minimal():
    packager = Packager()
    session = CaptureSession(
        session_id="S1",
        product_id="P1",
        operator_id="O1",
        status=AssetStatus.CAPTURED
    )
    
    frames = ["frame_001.jpg", "frame_002.jpg"]
    quality_report = {"blur_ok": True}
    coverage_report = {"overall_status": "sufficient"}
    
    draft = packager.prepare_reconstruction_payload(
        session, frames, quality_report, coverage_report
    )
    
    assert draft.job_id == "RJ-S1"
    assert draft.capture_session_id == "S1"
    assert draft.input_frames == frames
    assert draft.quality_report == quality_report
    assert draft.coverage_report == coverage_report
    assert draft.product_id == "P1"
    assert isinstance(draft.created_at, datetime)
