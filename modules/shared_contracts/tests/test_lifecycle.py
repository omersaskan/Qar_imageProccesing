import pytest
from shared_contracts.lifecycle import AssetStatus, can_transition, assert_transition
from shared_contracts.errors import InvalidTransitionError

def test_status_names():
    assert AssetStatus.CREATED == "created"
    assert AssetStatus.RECAPTURE_REQUIRED == "recapture_required"
    assert AssetStatus.PUBLISHED == "published"

def test_valid_transitions():
    assert can_transition(AssetStatus.CREATED, AssetStatus.CAPTURED) is True
    assert can_transition(AssetStatus.CREATED, AssetStatus.RECAPTURE_REQUIRED) is True
    assert can_transition(AssetStatus.CAPTURED, AssetStatus.RECONSTRUCTED) is True
    assert can_transition(AssetStatus.CAPTURED, AssetStatus.RECAPTURE_REQUIRED) is True
    assert can_transition(AssetStatus.PUBLISHED, AssetStatus.VALIDATED) is True

def test_invalid_transitions():
    assert can_transition(AssetStatus.CREATED, AssetStatus.PUBLISHED) is False
    assert can_transition(AssetStatus.PUBLISHED, AssetStatus.CREATED) is False

def test_assert_transition_success():
    # Should not raise exception
    assert_transition(AssetStatus.CREATED, AssetStatus.CAPTURED)

def test_assert_transition_failure():
    with pytest.raises(InvalidTransitionError) as excinfo:
        assert_transition(AssetStatus.CREATED, AssetStatus.PUBLISHED)
    assert "Invalid transition from 'created' to 'published'" in str(excinfo.value)
