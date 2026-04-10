import json
from pathlib import Path
from typing import Optional, Any
from modules.shared_contracts.models import CaptureSession, SessionEvent
from modules.shared_contracts.lifecycle import AssetStatus, assert_transition
from modules.utils.path_safety import validate_safe_path, ensure_dir, validate_identifier
from modules.utils.file_persistence import atomic_write_json, FileLock

class SessionManager:
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root).resolve()
        self.sessions_dir = self.data_root / "sessions"
        self.captures_dir = self.data_root / "captures"
        
        # Ensure directories exist
        ensure_dir(self.sessions_dir)
        ensure_dir(self.captures_dir)

    def create_session(self, session_id: str, product_id: str, operator_id: str) -> CaptureSession:
        session_id = validate_identifier(session_id, "Session ID")
        product_id = validate_identifier(product_id, "Product ID")
        
        session = CaptureSession(
            session_id=session_id,
            product_id=product_id,
            operator_id=operator_id,
            status=AssetStatus.CREATED,
            last_pipeline_stage=AssetStatus.CREATED.value,
        )
        
        # Setup capture folder structure
        session_capture_dir = validate_safe_path(self.captures_dir, session_id)
        ensure_dir(session_capture_dir / "video")
        ensure_dir(session_capture_dir / "frames")
        ensure_dir(session_capture_dir / "reports")
        
        # Record initial event
        session.history.append(SessionEvent(
            from_status="none",
            to_status=AssetStatus.CREATED.value,
            note="Session created"
        ))

        self.save_session(session)
        return session

    def _save_no_lock(self, session: CaptureSession) -> None:
        """Internal save without locking (caller must hold the lock)."""
        file_path = self.sessions_dir / f"{session.session_id}.json"
        atomic_write_json(file_path, session.model_dump(mode="json"))

    def save_session(self, session: CaptureSession) -> None:
        """Public save with locking."""
        file_path = self.sessions_dir / f"{session.session_id}.json"
        with FileLock(file_path):
            self._save_no_lock(session)

    def get_session(self, session_id: str) -> Optional[CaptureSession]:
        file_path = self.sessions_dir / f"{session_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return CaptureSession.model_validate(data)

    def update_session_status(self, session_id: str, new_status: AssetStatus) -> CaptureSession:
        return self.update_session(session_id, new_status=new_status)

    def update_session(self, session_id: str, new_status: Optional[AssetStatus] = None, **fields: Any) -> CaptureSession:
        file_path = self.sessions_dir / f"{session_id}.json"
        with FileLock(file_path):
            session = self.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")

            old_status = session.status
            if new_status is not None:
                assert_transition(old_status, new_status)
                session.status = new_status
                
                # Record event
                event = SessionEvent(
                    from_status=old_status.value,
                    to_status=new_status.value,
                    note=fields.get("failure_reason") or fields.get("note") or f"Transition to {new_status.value}",
                    stage=fields.get("last_pipeline_stage") or session.last_pipeline_stage
                )
                session.history.append(event)

            for field_name, value in fields.items():
                if field_name == "note": continue # internal metadata
                if not hasattr(session, field_name):
                    raise AttributeError(f"CaptureSession has no field '{field_name}'")
                setattr(session, field_name, value)

            self._save_no_lock(session)
            return session
    
    def get_capture_path(self, session_id: str) -> Path:
        return self.captures_dir / session_id
