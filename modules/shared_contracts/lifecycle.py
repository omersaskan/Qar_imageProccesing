from enum import Enum
from typing import Set, Dict
from .errors import InvalidTransitionError

class AssetStatus(str, Enum):
    CREATED = "created"
    CAPTURED = "captured"
    RECAPTURE_REQUIRED = "recapture_required"
    RECONSTRUCTED = "reconstructed"
    CLEANED = "cleaned"
    EXPORTED = "exported"
    VALIDATED = "validated"
    PUBLISHED = "published"
    FAILED = "failed"

class ReconstructionStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Define allowed transitions
_ALLOWED_TRANSITIONS: Dict[AssetStatus, Set[AssetStatus]] = {
    AssetStatus.CREATED: {AssetStatus.CAPTURED, AssetStatus.RECAPTURE_REQUIRED, AssetStatus.FAILED}, # Allow initial failure
    AssetStatus.CAPTURED: {AssetStatus.RECONSTRUCTED, AssetStatus.CAPTURED, AssetStatus.RECAPTURE_REQUIRED, AssetStatus.FAILED}, # Retry capture or fail
    AssetStatus.RECAPTURE_REQUIRED: {AssetStatus.CAPTURED, AssetStatus.FAILED}, # Recapture can unblock the session
    AssetStatus.RECONSTRUCTED: {AssetStatus.CLEANED, AssetStatus.CAPTURED, AssetStatus.FAILED}, # Redo capture or fail
    AssetStatus.CLEANED: {AssetStatus.EXPORTED, AssetStatus.RECONSTRUCTED, AssetStatus.FAILED}, # Re-export, redo reconstruction or fail
    AssetStatus.EXPORTED: {AssetStatus.VALIDATED, AssetStatus.CLEANED, AssetStatus.FAILED}, # Revalidate, redo cleanup or fail
    AssetStatus.VALIDATED: {AssetStatus.PUBLISHED, AssetStatus.EXPORTED, AssetStatus.FAILED}, # Publish, re-export or fail
    AssetStatus.PUBLISHED: {AssetStatus.VALIDATED}, # Rollback to validated state
    AssetStatus.FAILED: set(), # Terminal state
}

def can_transition(from_status: AssetStatus, to_status: AssetStatus) -> bool:
    """Check if a transition from from_status to to_status is allowed."""
    return to_status in _ALLOWED_TRANSITIONS.get(from_status, set())

def assert_transition(from_status: AssetStatus, to_status: AssetStatus) -> None:
    """Assert that a transition is allowed, raising InvalidTransitionError if not."""
    if not can_transition(from_status, to_status):
        raise InvalidTransitionError(from_status.value, to_status.value)
