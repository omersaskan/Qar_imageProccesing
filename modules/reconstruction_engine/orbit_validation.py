"""
Orbit validation — Sprint 5.

Compares pose-backed coverage metrics against Sprint 2 heuristic thresholds
and emits a structured verdict.

Default thresholds (all opt-in configurable):
  - min_coverage_ratio: 0.40  (≥40% of 3×8 grid cells occupied)
  - min_azimuth_span:   270   (≥270° arc around subject)
  - min_elevation_spread: 30  (≥30° vertical spread)
  - min_registered_ratio: 0.70 (≥70% of input frames registered by COLMAP)

Verdict levels: pass | review | fail | unavailable
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OrbitThresholds:
    min_coverage_ratio: float = 0.40
    min_azimuth_span_degrees: float = 270.0
    min_elevation_spread_degrees: float = 30.0
    min_registered_ratio: float = 0.70


@dataclass
class OrbitValidationResult:
    verdict: str = "unavailable"   # pass | review | fail | unavailable
    reasons: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    coverage_ratio: float = 0.0
    azimuth_span_degrees: float = 0.0
    elevation_spread_degrees: float = 0.0
    registered_ratio: Optional[float] = None
    thresholds: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_orbit(
    coverage: Dict[str, Any],
    total_input_frames: int = 0,
    thresholds: Optional[OrbitThresholds] = None,
) -> OrbitValidationResult:
    """
    Validate orbit quality from a pose_coverage_matrix result dict.

    coverage: output of pose_coverage_matrix.build_coverage_matrix()
    total_input_frames: number of keyframes fed into COLMAP (for registered_ratio)
    """
    th = thresholds or OrbitThresholds()

    if coverage.get("status") != "ok":
        return OrbitValidationResult(
            verdict="unavailable",
            reasons=[coverage.get("reason", "pose coverage unavailable")],
            thresholds=asdict(th),
        )

    cov_ratio = coverage.get("coverage_ratio", 0.0)
    az_span = coverage.get("azimuth_span_degrees", 0.0)
    el_spread = coverage.get("elevation_spread_degrees", 0.0)
    registered = coverage.get("registered_count", 0)

    reg_ratio: Optional[float] = None
    if total_input_frames > 0:
        reg_ratio = registered / total_input_frames

    reasons: List[str] = []
    suggestions: List[str] = []
    fail_flags: List[bool] = []
    review_flags: List[bool] = []

    # Coverage ratio
    if cov_ratio < th.min_coverage_ratio:
        msg = f"coverage_ratio={cov_ratio:.2f} < threshold={th.min_coverage_ratio}"
        if cov_ratio < th.min_coverage_ratio * 0.6:
            reasons.append(msg)
            fail_flags.append(True)
            suggestions.append("Capture more angles around the subject")
        else:
            reasons.append(msg)
            review_flags.append(True)
            suggestions.append("Coverage is marginal; consider additional orbit passes")

    # Azimuth span
    if az_span < th.min_azimuth_span_degrees:
        msg = f"azimuth_span={az_span:.1f}° < threshold={th.min_azimuth_span_degrees}°"
        if az_span < th.min_azimuth_span_degrees * 0.70:
            reasons.append(msg)
            fail_flags.append(True)
            suggestions.append("Incomplete orbit — capture a full 360° pass around the object")
        else:
            reasons.append(msg)
            review_flags.append(True)

    # Elevation spread
    if el_spread < th.min_elevation_spread_degrees:
        msg = f"elevation_spread={el_spread:.1f}° < threshold={th.min_elevation_spread_degrees}°"
        reasons.append(msg)
        review_flags.append(True)
        suggestions.append("Add top-down shots for better vertical coverage")

    # Registered ratio
    if reg_ratio is not None and reg_ratio < th.min_registered_ratio:
        msg = f"registered_ratio={reg_ratio:.2f} < threshold={th.min_registered_ratio}"
        if reg_ratio < th.min_registered_ratio * 0.6:
            reasons.append(msg)
            fail_flags.append(True)
            suggestions.append("Too many frames failed to register — check blur / lighting")
        else:
            reasons.append(msg)
            review_flags.append(True)

    if fail_flags:
        verdict = "fail"
    elif review_flags:
        verdict = "review"
    else:
        verdict = "pass"

    return OrbitValidationResult(
        verdict=verdict,
        reasons=reasons,
        suggestions=suggestions,
        coverage_ratio=cov_ratio,
        azimuth_span_degrees=az_span,
        elevation_spread_degrees=el_spread,
        registered_ratio=reg_ratio,
        thresholds=asdict(th),
    )
