from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from modules.capture_workflow.coverage_analyzer import CoverageAnalyzer
from modules.capture_workflow.config import CoverageConfig


def _write_frame_with_mask(frame_path: Path, center_x: int, radius: int, color_shift: int = 0, legacy_naming: bool = True) -> None:
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    masks_dir = frame_path.parent / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    frame[:] = (20 + color_shift, 25 + color_shift, 30 + color_shift)
    mask = np.zeros((320, 320), dtype=np.uint8)
    
    cv2.circle(frame, (center_x, 160), radius, (180, 180, 200), -1)
    cv2.circle(mask, (center_x, 160), radius, 255, -1)
    
    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(frame_path)
    
    if legacy_naming:
        # frame_0000.jpg.png
        Image.fromarray(mask).save(masks_dir / f"{frame_path.name}.png")
    else:
        # frame_0000.png
        Image.fromarray(mask).save(masks_dir / f"{frame_path.stem}.png")


def test_coverage_sufficient(tmp_path):
    analyzer = CoverageAnalyzer()
    frames = []

    centers = [90, 110, 130, 160, 190, 220, 240, 210, 170, 130]
    radii = [48, 54, 60, 64, 60, 54, 48, 58, 66, 52]
    for index, (center, radius) in enumerate(zip(centers, radii)):
        frame_path = tmp_path / f"frame_{index:04d}.jpg"
        _write_frame_with_mask(frame_path, center, radius, color_shift=index * 3)
        frames.append(str(frame_path))

    report = analyzer.analyze_coverage(frames)

    assert report["num_frames"] == 10
    assert report["readable_frames"] == 10
    assert report["unique_views"] >= 5
    assert report["diversity"] == "sufficient"
    assert report["overall_status"] == "sufficient"
    assert report["coverage_score"] > 0.70


def test_coverage_insufficient_for_redundant_views(tmp_path):
    analyzer = CoverageAnalyzer()
    frames = []

    for index in range(5):
        frame_path = tmp_path / f"frame_{index:04d}.jpg"
        _write_frame_with_mask(frame_path, center_x=160, radius=58, color_shift=0)
        frames.append(str(frame_path))

    report = analyzer.analyze_coverage(frames)

    assert report["num_frames"] == 5
    assert report["overall_status"] == "insufficient"
    assert report["unique_views"] < 6
    assert report["recommended_action"] == "needs_recapture"
    assert any("viewpoint diversity" in reason.lower() for reason in report["reasons"])


def test_coverage_thresholds_are_configurable(tmp_path):
    analyzer = CoverageAnalyzer(
        CoverageConfig(
            min_readable_frames=4,
            min_unique_views=1,
            min_center_x_span=0.02,
            min_center_y_span=0.02,
            min_scale_variation=1.02,
            min_aspect_variation=0.01,
        )
    )
    frames = []

    for index, center in enumerate((150, 160, 170, 180)):
        frame_path = tmp_path / f"frame_{index:04d}.jpg"
        _write_frame_with_mask(frame_path, center_x=center, radius=60 + index, color_shift=index * 5)
        frames.append(str(frame_path))

    report = analyzer.analyze_coverage(frames)

    assert report["overall_status"] == "sufficient"
    assert report["recommended_action"] == "reconstruct"


def test_coverage_supports_stem_based_paths(tmp_path):
    analyzer = CoverageAnalyzer()
    frames = []

    for index in range(5):
        frame_path = tmp_path / f"frame_{index:04d}.jpg"
        # Explicitly use stem-based naming (frame_0000.png)
        _write_frame_with_mask(frame_path, center_x=160, radius=58, color_shift=0, legacy_naming=False)
        frames.append(str(frame_path))

    report = analyzer.analyze_coverage(frames)

    # We expect readable_frames to be 5, proving stem-based resolution worked
    assert report["num_frames"] == 5
    assert report["readable_frames"] == 5

def test_coverage_good_centered_orbit_is_sufficient(tmp_path):
    """
    Test that a well-distributed orbit (good number of frames, many unique views) 
    that happens to stay relatively centered within the frame span is not rejected as false-positive recapture.
    """
    analyzer = CoverageAnalyzer()
    frames = []

    # Generates many frames but with smaller relative x span (centered orbit)
    # The default min_viewpoint_spread is now 0.25 (80 pixels on 320 width)
    # To get 5 unique views with 28.8px dedupe distance, we need a larger span than 70.
    # [100, 130, 160, 190, 220, 240] are spaced enough.
    centers = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
    radii = [60] * len(centers)
    for index, (center, radius) in enumerate(zip(centers, radii)):
        frame_path = tmp_path / f"frame_{index:04d}.jpg"
        _write_frame_with_mask(frame_path, center_x=center, radius=radius, color_shift=index * 4)
        frames.append(str(frame_path))

    report = analyzer.analyze_coverage(frames)

    assert report["num_frames"] == len(centers)
    assert report["overall_status"] == "sufficient"
    assert report["recommended_action"] == "reconstruct"
    
    # Check that hard reasons does not include low coverage
    assert not any("LOW_HORIZONTAL_COVERAGE" in reason for reason in report.get("hard_reasons", []))
