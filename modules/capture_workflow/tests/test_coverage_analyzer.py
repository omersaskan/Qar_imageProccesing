import pytest
from capture_workflow.coverage_analyzer import CoverageAnalyzer

def test_coverage_sufficient():
    analyzer = CoverageAnalyzer()
    # 35 frames should be sufficient
    frames = [f"frame_{i}.jpg" for i in range(35)]
    report = analyzer.analyze_coverage(frames)
    
    assert report["num_frames"] == 35
    assert report["diversity"] == "sufficient"
    assert report["top_down_captured"] is True
    assert report["overall_status"] == "sufficient"
    assert len(report["reasons"]) == 0

def test_coverage_insufficient():
    analyzer = CoverageAnalyzer()
    # 5 frames should be insufficient
    frames = [f"frame_{i}.jpg" for i in range(5)]
    report = analyzer.analyze_coverage(frames)
    
    assert report["num_frames"] == 5
    assert report["diversity"] == "insufficient"
    assert report["top_down_captured"] is False
    assert report["overall_status"] == "insufficient"
    assert len(report["reasons"]) > 0
    assert "Too few high-quality frames extracted." in report["reasons"]
