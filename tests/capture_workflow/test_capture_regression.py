import pytest
from pathlib import Path
from modules.capture_workflow.coverage_analyzer import CoverageAnalyzer
from modules.capture_workflow.frame_extractor import FrameExtractor

def test_coverage_analyzer_instantiation():
    analyzer = CoverageAnalyzer()
    assert analyzer is not None
    assert analyzer.config.min_readable_frames > 0

def test_frame_extractor_instantiation():
    extractor = FrameExtractor()
    assert extractor is not None
    assert extractor.config.min_frames >= 3
