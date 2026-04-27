
import os
import pytest
from unittest.mock import patch, MagicMock
from modules.capture_workflow.object_masker import ObjectMasker
from modules.capture_workflow.config import default_segmentation_config
import numpy as np

def test_object_masker_fallback_to_legacy():
    """Verify that ObjectMasker falls back to legacy if SAM2 is requested but HAS_SAM2 is False."""
    
    # Mock HAS_SAM2 to be False
    with patch("modules.ai_segmentation.sam2_wrapper.HAS_SAM2", False), \
         patch.dict(os.environ, {"SEGMENTATION_METHOD": "sam2"}):
        
        masker = ObjectMasker()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # We need to mock BackendFactory.get_backend to see what was called
        with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock_get_backend:
            # Mock a backend to return something valid
            mock_backend = MagicMock()
            mock_backend.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {})
            mock_get_backend.return_value = mock_backend
            
            masker.generate_mask(frame)
            
            # Should have called get_backend with 'heuristic' (default fallback) or 'rembg' (default config)
            # but NOT 'sam2' because HAS_SAM2 is False
            called_names = [call.args[0] for call in mock_get_backend.call_args_list]
            assert "sam2" not in called_names

def test_object_masker_uses_sam2_when_enabled():
    """Verify that ObjectMasker attempts to use sam2 backend when flag is set and HAS_SAM2 is True."""
    
    with patch("modules.ai_segmentation.sam2_wrapper.HAS_SAM2", True), \
         patch.dict(os.environ, {"SEGMENTATION_METHOD": "sam2"}):
        
        masker = ObjectMasker()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with patch("modules.capture_workflow.segmentation_backends.factory.BackendFactory.get_backend") as mock_get_backend:
            mock_backend = MagicMock()
            mock_backend.segment.return_value = (np.zeros((100, 100), dtype=np.uint8), {})
            mock_get_backend.return_value = mock_backend
            
            masker.generate_mask(frame)
            
            called_names = [call.args[0] for call in mock_get_backend.call_args_list]
            assert "sam2" in called_names
