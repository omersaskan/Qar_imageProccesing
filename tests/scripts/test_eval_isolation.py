"""
Tests for run_sam2_dev_subset.py Isolation
===========================================
Ensures that the evaluation script correctly isolates legacy vs SAM2 phases
even when SAM2 is enabled globally.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import cv2
import json

from scripts.run_sam2_dev_subset import run_evaluation
from modules.operations.settings import settings

class TestEvalIsolation:
    @patch("modules.capture_workflow.object_masker.ObjectMasker")
    @patch("modules.ai_segmentation.sam2_wrapper.SAM2Wrapper")
    def test_legacy_phase_uses_legacy_only(self, mock_sam2, mock_masker, tmp_path):
        """
        Even if SAM2 is enabled globally, the legacy phase of evaluation 
        must force settings.sam2_enabled = False and use the legacy backend.
        """
        # Setup paths
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        frame_path = frames_dir / "frame_0000.jpg"
        cv2.imwrite(str(frame_path), np.zeros((100, 100, 3), dtype=np.uint8))
        
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        
        output_dir = tmp_path / "results"
        
        # Mock ObjectMasker
        mock_masker_inst = mock_masker.return_value
        mock_masker_inst.generate_mask.return_value = (
            np.zeros((100, 100), dtype=np.uint8),
            {"backend_name": "heuristic", "ai_segmentation_used": False}
        )
        
        # Mock SAM2 to be unavailable initially
        mock_sam2_inst = mock_sam2.return_value
        mock_sam2_inst.is_available.return_value = False
        mock_sam2_inst.get_status.return_value = {"sam2_available": False}
        
        # Enable SAM2 globally in settings
        settings.sam2_enabled = True
        settings.segmentation_method = "sam2"
            
        args = MagicMock()
        args.frames_dir = str(frames_dir)
        args.gt_dir = str(gt_dir)
        args.output_dir = str(output_dir)
        args.capture_id = None
        
        results = run_evaluation(args)
        
        # Verify results
        assert results["verification"]["legacy_method_verified"] is True
        assert results["verification"]["legacy_ai_segmentation_used"] is False
        assert "heuristic" in results["verification"]["legacy_backends_detected"]
        
        # Verify settings were restored (actually they weren't changed permanently 
        # but patch.object handles it. But here I changed them manually before run)
        # and run_evaluation doesn't restore them if it didn't use finally on them?
        # Wait, run_evaluation uses patch.object internally now.
        
        # After run_evaluation, settings should be back to what they were before run_evaluation
        # Wait, run_evaluation uses patch.object internally, so it only affects the 'with' block.
        # My manual change above stays.
        assert settings.sam2_enabled is True
        assert settings.segmentation_method == "sam2"

    @patch("modules.capture_workflow.object_masker.ObjectMasker")
    @patch("modules.ai_segmentation.sam2_wrapper.SAM2Wrapper")
    def test_sam2_phase_uses_sam2(self, mock_sam2, mock_masker, tmp_path):
        """
        Verify that SAM2 phase uses SAM2 even if globally it might be legacy.
        """
        # Setup paths
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        frame_path = frames_dir / "frame_0000.jpg"
        cv2.imwrite(str(frame_path), np.zeros((100, 100, 3), dtype=np.uint8))
        
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        
        output_dir = tmp_path / "results"
        
        # Mock legacy phase
        mock_masker_inst = mock_masker.return_value
        mock_masker_inst.generate_mask.return_value = (
            np.zeros((100, 100), dtype=np.uint8),
            {"backend_name": "heuristic", "ai_segmentation_used": False}
        )
        
        # Mock SAM2 to be available
        mock_sam2_inst = mock_sam2.return_value
        mock_sam2_inst.is_available.return_value = True
        mock_sam2_inst.get_status.return_value = {"sam2_available": True, "sam2_model_loaded": True}
        mock_sam2_inst.segment_frame.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        # Disable SAM2 globally
        settings.sam2_enabled = False
        settings.segmentation_method = "legacy"
            
        args = MagicMock()
        args.frames_dir = str(frames_dir)
        args.gt_dir = str(gt_dir)
        args.output_dir = str(output_dir)
        args.capture_id = None
        
        results = run_evaluation(args)
        
        # Verify SAM2 was run
        assert results["sam2_ran"] is True
        assert results["verification"]["sam2_method_verified"] is True
        assert results["verification"]["sam2_ai_segmentation_used"] is True
        assert "sam2" in results["verification"]["sam2_backends_detected"]
        
        # Verify settings were restored to what they were before run
        assert settings.sam2_enabled is False
        assert settings.segmentation_method == "legacy"
