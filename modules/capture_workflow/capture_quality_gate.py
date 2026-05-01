"""
Capture Quality Gate v2 — orchestrator combining blur burst + elevation
distribution + azimuth diversity into a single decision the API and UI can act on.

Decision tiers:
    - "pass"        — capture is solid; reconstruction can proceed
    - "review"      — proceed but flag for QA (operator may want a re-take)
    - "reshoot"     — strongly recommend re-capture; reconstruction will likely
                       waste compute

The gate is *advisory* until Sprint 2 wires it into the upload preflight as
a hard reject (currently it returns suggestions in the API response).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .azimuth_diversity import AzimuthReport, estimate_azimuth_distribution
from .blur_burst_detector import BlurBurstReport, detect_bursts
from .elevation_estimator import ElevationReport, estimate_elevation_distribution


@dataclass
class GateThresholds:
    burst_ratio_warn: float = 0.10           # ≥10% frames in bursts
    burst_ratio_fail: float = 0.25           # ≥25% — recapture
    multi_height_warn: float = 0.34          # only one elevation band → warn
    multi_height_fail: float = 0.10          # essentially no elevation diversity
    azimuth_orbit_warn: float = 0.40         # cumulative motion below
    azimuth_orbit_fail: float = 0.20         # ≈ fixed viewpoint
    static_run_ratio_warn: float = 0.30      # ≥30% of frames static run
    min_frame_count_for_gate: int = 8        # below this, results are not statistically meaningful


@dataclass
class CaptureGateReport:
    decision: str                            # "pass" | "review" | "reshoot"
    reasons: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    blur: Dict[str, Any] = field(default_factory=dict)
    elevation: Dict[str, Any] = field(default_factory=dict)
    azimuth: Dict[str, Any] = field(default_factory=dict)
    matrix_3x8: List[List[int]] = field(default_factory=list)  # rows: low/mid/top, cols: 8 azimuth buckets
    gate_thresholds: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_3x8_matrix(elevation: ElevationReport, azimuth: AzimuthReport) -> List[List[int]]:
    """
    Visual placeholder matrix for the UI overlay.
    Until real per-frame yaw exists we approximate column distribution
    by spreading frames evenly across azimuth buckets, weighted by
    cumulative_orbit_progress.
    """
    matrix = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(3)]  # rows: low / mid / top
    if elevation.frame_count == 0:
        return matrix

    bucket_index = {"low": 0, "mid": 1, "top": 2, "unknown": 1}

    # If we have orbit progress, sweep cells from col 0..N
    azimuth_span = max(1, int(round(azimuth.estimated_coverage_ratio * 8)))
    spread_columns = list(range(azimuth_span))

    for fe in elevation.per_frame:
        row = bucket_index.get(fe.bucket, 1)
        col = spread_columns[hash(fe.frame_name) % len(spread_columns)] if spread_columns else 0
        matrix[row][col] += 1
    return matrix


def evaluate_capture(
    frame_paths: List[str],
    masks_dir: Optional[Path] = None,
    thresholds: Optional[GateThresholds] = None,
) -> CaptureGateReport:
    """
    Run all 3 sub-detectors and aggregate into a single decision.
    Inputs are file paths (frames already extracted).  Safe on small or
    pathological input — returns a graded `reshoot` rather than crashing.
    """
    th = thresholds or GateThresholds()
    rep = CaptureGateReport(decision="pass", gate_thresholds=asdict(th))
    # Always ship a 3×8 skeleton so the UI never has to defensive-code shape.
    rep.matrix_3x8 = [[0] * 8 for _ in range(3)]

    if not frame_paths:
        rep.decision = "reshoot"
        rep.reasons.append("no frames extracted")
        rep.suggestions.append("re-capture with sufficient duration / coverage")
        return rep

    if len(frame_paths) < th.min_frame_count_for_gate:
        rep.decision = "review"
        rep.reasons.append(f"only {len(frame_paths)} frames < gate minimum {th.min_frame_count_for_gate}")
        rep.suggestions.append("longer capture for statistically meaningful coverage")

    # 1. Blur bursts
    blur_rep = detect_bursts(frame_paths)
    rep.blur = blur_rep.to_dict()
    if blur_rep.burst_ratio >= th.burst_ratio_fail:
        rep.decision = _worse(rep.decision, "reshoot")
        rep.reasons.append(
            f"blur_burst_ratio {blur_rep.burst_ratio:.0%} >= {th.burst_ratio_fail:.0%}"
        )
        rep.suggestions.append("hold the camera steadier; avoid fast pans during capture")
    elif blur_rep.burst_ratio >= th.burst_ratio_warn:
        rep.decision = _worse(rep.decision, "review")
        rep.reasons.append(
            f"blur_burst_ratio {blur_rep.burst_ratio:.0%} between warn & fail"
        )

    # 2. Elevation distribution
    elev_rep = estimate_elevation_distribution(frame_paths, masks_dir=masks_dir)
    rep.elevation = elev_rep.to_dict()
    if elev_rep.multi_height_score < th.multi_height_fail:
        rep.decision = _worse(rep.decision, "reshoot")
        rep.reasons.append(
            f"multi_height_score {elev_rep.multi_height_score:.2f} < {th.multi_height_fail:.2f}"
        )
        rep.suggestions.append("capture from low / eye-level / top-down — three height passes")
    elif elev_rep.multi_height_score < th.multi_height_warn:
        rep.decision = _worse(rep.decision, "review")
        rep.reasons.append(
            f"multi_height_score {elev_rep.multi_height_score:.2f} below warn threshold"
        )
        rep.suggestions.append("add at least one extra elevation band (top-down or low-angle)")

    # 3. Azimuth diversity
    az_rep = estimate_azimuth_distribution(frame_paths, masks_dir=masks_dir)
    rep.azimuth = az_rep.to_dict()
    if az_rep.cumulative_orbit_progress < th.azimuth_orbit_fail:
        rep.decision = _worse(rep.decision, "reshoot")
        rep.reasons.append(
            f"orbit_progress {az_rep.cumulative_orbit_progress:.2f} < {th.azimuth_orbit_fail:.2f}"
        )
        rep.suggestions.append("walk around the object — at least 270° orbit")
    elif az_rep.cumulative_orbit_progress < th.azimuth_orbit_warn:
        rep.decision = _worse(rep.decision, "review")
        rep.reasons.append(
            f"orbit_progress {az_rep.cumulative_orbit_progress:.2f} below warn threshold"
        )

    if (az_rep.max_consecutive_static_frames > 0 and
            az_rep.frame_count > 0 and
            az_rep.max_consecutive_static_frames / max(az_rep.frame_count, 1) >= th.static_run_ratio_warn):
        rep.decision = _worse(rep.decision, "review")
        rep.reasons.append(
            f"static-run {az_rep.max_consecutive_static_frames}/{az_rep.frame_count} frames"
        )
        rep.suggestions.append("avoid pausing — keep moving around the object")

    # 4. Build the 3×8 visual matrix for the UI
    rep.matrix_3x8 = _build_3x8_matrix(elev_rep, az_rep)

    if not rep.reasons:
        rep.notes.append("all sub-detectors within thresholds")

    return rep


_RANK = {"pass": 0, "review": 1, "reshoot": 2}


def _worse(current: str, candidate: str) -> str:
    return candidate if _RANK.get(candidate, 0) > _RANK.get(current, 0) else current
