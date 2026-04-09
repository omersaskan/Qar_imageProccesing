import cv2
import numpy as np

from modules.capture_workflow.object_masker import ObjectMasker
from modules.capture_workflow.quality_analyzer import QualityAnalyzer


def _make_gradient_frame(size: int = 400) -> np.ndarray:
    x = np.linspace(20, 36, size, dtype=np.uint8)
    y = np.linspace(24, 40, size, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy, np.full_like(xx, 30)], axis=2)


def test_mask_metrics_on_synthetic():
    masker = ObjectMasker()
    analyzer = QualityAnalyzer()

    frame = _make_gradient_frame()
    cv2.circle(frame, (200, 200), 100, (200, 200, 200), -1)
    for x, y in ((170, 170), (210, 185), (230, 210), (180, 235), (220, 245)):
        cv2.circle(frame, (x, y), 4, (110, 110, 110), -1)

    mask, meta = masker.generate_mask(frame)
    analysis = analyzer.analyze_frame(frame, mask, meta)

    assert meta["confidence"] > 0.55
    assert meta["solidity"] > 0.90
    assert meta["fragment_count"] == 1
    assert meta["purity_score"] > 0.60
    assert analysis["overall_pass"] is True
    assert analysis["mask_confidence"] > 0.55


def test_bottom_support_detection_rejects_contaminated_frame():
    masker = ObjectMasker()
    analyzer = QualityAnalyzer()

    frame = np.zeros((420, 420, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 300), (419, 380), (90, 90, 90), -1)
    cv2.circle(frame, (210, 220), 85, (210, 210, 210), -1)

    mask, meta = masker.generate_mask(frame)
    analysis = analyzer.analyze_frame(frame, mask, meta)

    assert meta["bottom_band_ratio"] >= 0.0
    assert meta["purity_score"] >= 0.0
    assert (
        meta["support_removed"]
        or meta["support_suspected"]
        or "bottom_support_band" in " ".join(analysis["failure_reasons"])
        or "support_contamination_detected" in analysis["failure_reasons"]
    )


def test_clipping_detection():
    masker = ObjectMasker()
    analyzer = QualityAnalyzer()

    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(frame, (0, 200), 80, (200, 200, 200), -1)

    mask, meta = masker.generate_mask(frame)
    analysis = analyzer.analyze_frame(frame, mask, meta)

    assert meta["is_clipped"] is True
    assert analysis["is_clipped"] is True
    assert "subject_clipped" in analysis["failure_reasons"]
