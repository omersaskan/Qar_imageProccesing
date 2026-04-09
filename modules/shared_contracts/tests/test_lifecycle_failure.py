import pytest
from modules.shared_contracts.lifecycle import AssetStatus, can_transition, assert_transition
from modules.shared_contracts.errors import InvalidTransitionError

def test_failed_status_is_reachable():
    # Reachable from every in-flight stage.
    assert can_transition(AssetStatus.CREATED, AssetStatus.FAILED)
    assert can_transition(AssetStatus.CAPTURED, AssetStatus.FAILED)
    assert can_transition(AssetStatus.RECONSTRUCTED, AssetStatus.FAILED)
    assert can_transition(AssetStatus.CLEANED, AssetStatus.FAILED)
    assert can_transition(AssetStatus.EXPORTED, AssetStatus.FAILED)
    assert can_transition(AssetStatus.VALIDATED, AssetStatus.FAILED)

def test_failed_status_is_terminal():
    # Should not be able to transition OUT of FAILED
    for status in AssetStatus:
        if status != AssetStatus.FAILED:
            assert not can_transition(AssetStatus.FAILED, status)

def test_invalid_transitions_raise_error():
    with pytest.raises(InvalidTransitionError):
        assert_transition(AssetStatus.FAILED, AssetStatus.CAPTURED)
    
    with pytest.raises(InvalidTransitionError):
        assert_transition(AssetStatus.CREATED, AssetStatus.PUBLISHED)
