
import pytest
from unittest.mock import patch, MagicMock
from modules.capture_workflow.object_masker import ObjectMasker
from modules.operations.settings import settings
import numpy as np

def test_object_masker_fallback_when_sam2_disabled():
    """Verify that ObjectMasker falls back to legacy if SEGMENTATION_METHOD=sam2 but SAM2_ENABLED=False."""
    
    # We patch settings directly or the SAM2Wrapper.is_available
    with patch("modules.operations.settings.settings.segmentation_method", "sam2"), \
         patch("modules.operations.settings.settings.sam2_enabled", False):
        
        masker = ObjectMasker()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {})
            mock_get_backend.return_value = mock_backend
            
            binary, meta = masker.generate_mask(frame)
            
            # Should NOT have called get_backend with 'sam2'
            called_names = [call.args[0] for call in mock_get_backend.call_args_list]
            assert "sam2" not in called_names
            assert meta.get("fallback_used") is True
            assert "SAM2 disabled" in meta.get("fallback_reason")

def test_object_masker_fallback_when_checkpoint_missing():
    """Verify that ObjectMasker falls back if SAM2_ENABLED=True but checkpoint is missing."""
    
    with patch("modules.operations.settings.settings.segmentation_method", "sam2"), \
         patch("modules.operations.settings.settings.sam2_enabled", True), \
         patch("modules.operations.settings.settings.sam2_checkpoint", "non_existent.pt"), \
         patch("pathlib.Path.exists", return_value=False):
        
        masker = ObjectMasker()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {})
            mock_get_backend.return_value = mock_backend
            
            binary, meta = masker.generate_mask(frame)
            
            assert "sam2" not in [call.args[0] for call in mock_get_backend.call_args_list]
            assert meta.get("fallback_used") is True
            assert "Checkpoint not found" in meta.get("fallback_reason")

def test_object_masker_uses_sam2_when_fully_enabled():
    """Verify that ObjectMasker uses sam2 when enabled and available."""
    
    # We need to mock HAS_SAM2 to be True for the import block in ObjectMasker/SAM2Wrapper
    with patch("modules.operations.settings.settings.segmentation_method", "sam2"), \
         patch("modules.operations.settings.settings.sam2_enabled", True), \
         patch("modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True), \
         patch("pathlib.Path.exists", return_value=True):
        
        masker = ObjectMasker()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {})
            mock_get_backend.return_value = mock_backend
            
            masker.generate_mask(frame)
            
            assert "sam2" in [call.args[0] for call in mock_get_backend.call_args_list]

def test_object_masker_handles_backend_not_implemented():
    """Verify that ObjectMasker falls back if SAM2 backend raises NotImplementedError."""
    
    with patch("modules.operations.settings.settings.segmentation_method", "sam2"), \
         patch("modules.operations.settings.settings.sam2_enabled", True), \
         patch("modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True), \
         patch("pathlib.Path.exists", return_value=True):
        
        masker = ObjectMasker()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock_get_backend:
            # First call returns SAM2 backend which fails
            sam2_backend = MagicMock()
            sam2_backend.segment.side_effect = NotImplementedError("Not yet")
            
            # Second call (fallback) returns heuristic
            heuristic_backend = MagicMock()
            heuristic_backend.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {"original": "meta"})
            
            mock_get_backend.side_effect = lambda name: sam2_backend if name == "sam2" else heuristic_backend
            
            binary, meta = masker.generate_mask(frame)
            
            assert meta.get("fallback_used") is True
            assert "Backend sam2 failed" in meta.get("fallback_reason")
            assert meta.get("requested_segmentation_method") == "sam2"
