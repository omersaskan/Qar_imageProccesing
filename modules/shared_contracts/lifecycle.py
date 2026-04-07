from enum import Enum
from typing import Set, Dict
from .errors import InvalidTransitionError

class AssetStatus(str, Enum):
    CREATED = "created"
    CAPTURED = "captured"
    RECONSTRUCTED = "reconstructed"
    CLEANED = "cleaned"
    EXPORTED = "exported"
    VALIDATED = "validated"
    PUBLISHED = "published"

class ReconstructionStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Define allowed transitions
_ALLOWED_TRANSITIONS: Dict[AssetStatus, Set[AssetStatus]] = {
    AssetStatus.CREATED: {AssetStatus.CAPTURED},
    AssetStatus.CAPTURED: {AssetStatus.RECONSTRUCTED, AssetStatus.CAPTURED}, # Retry capture
    AssetStatus.RECONSTRUCTED: {AssetStatus.CLEANED, AssetStatus.CAPTURED}, # Fail/Retry capture
    AssetStatus.CLEANED: {AssetStatus.EXPORTED, AssetStatus.RECONSTRUCTED}, # Redo cleanup or redo reconstruction
    AssetStatus.EXPORTED: {AssetStatus.VALIDATED, AssetStatus.CLEANED}, # Re-export or redo cleanup
    AssetStatus.VALIDATED: {AssetStatus.PUBLISHED, AssetStatus.EXPORTED}, # Publish or re-export
    AssetStatus.PUBLISHED: {AssetStatus.VALIDATED}, # Rollback to validated state
}

def can_transition(from_status: AssetStatus, to_status: AssetStatus) -> bool:
    """Check if a transition from from_status to to_status is allowed."""
    return to_status in _ALLOWED_TRANSITIONS.get(from_status, set())

def assert_transition(from_status: AssetStatus, to_status: AssetStatus) -> None:
    """Assert that a transition is allowed, raising InvalidTransitionError if not."""
    if not can_transition(from_status, to_status):
        raise InvalidTransitionError(from_status.value, to_status.value)
