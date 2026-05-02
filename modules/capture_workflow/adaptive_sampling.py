"""
Adaptive temporal sampler — replaces the fixed `frame_sample_rate` with
motion-aware decimation.

Why fixed N-th frame is wrong:
    - Fast pan + sample_rate=5 → 30 fps source = blur burst, every 6th
      frame still in motion zone.
    - Slow careful capture + sample_rate=5 → 30 fps source = redundant
      near-identical frames.

Heuristic per-frame decision (cheap):
    1. Sparse optical flow (Lucas-Kanade) on Shi-Tomasi corners between
       prev_frame and curr_frame → median magnitude.
    2. Mask bbox IoU vs last kept frame's bbox.
    3. Sharpness (Laplacian variance, reused from blur_burst_detector).

Decision tree:
    - too static (flow < min_flow AND IoU > 0.95) → skip_static
    - too blurry (sharpness < min_sharpness) → skip_blurry
    - too redundant (IoU > redundant_iou AND flow < redundant_flow) → skip_redundant
    - motion-burst (flow > burst_flow) → tag with motion_burst (caller may
      decide to keep/drop based on coverage)
    - else → keep

Outputs are advisory; the caller (frame_extractor or coverage-aware selector)
makes the final keep/skip decision.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class SamplingVerdict(str, Enum):
    KEEP = "keep"
    SKIP_STATIC = "skip_static"
    SKIP_REDUNDANT = "skip_redundant"
    SKIP_BLURRY = "skip_blurry"
    KEEP_MOTION_BURST = "keep_motion_burst"   # tagged for caller policy
    KEEP_FORCED = "keep_forced"               # first frame, periodic anchor


@dataclass
class SamplingThresholds:
    # Optical flow median magnitude in pixels per frame
    min_flow_static: float = 1.5
    redundant_flow: float = 4.0
    burst_flow: float = 35.0

    # bbox IoU vs last kept frame
    redundant_iou: float = 0.92
    static_iou: float = 0.97

    # Sharpness (Laplacian variance, downsampled grayscale)
    min_sharpness: float = 60.0

    # Force-keep cadence — emit a frame at least every N raw frames
    # so a perfectly static scene still produces something.
    force_keep_every_n_raw_frames: int = 60

    # Adaptive baseline — minimum gap (raw frames) between two kept frames
    # so blur-burst detection has a chance to fire.
    min_gap_raw_frames: int = 1


@dataclass
class SamplingDecision:
    verdict: SamplingVerdict
    reasons: List[str] = field(default_factory=list)
    flow_median: float = 0.0
    bbox_iou: float = 0.0
    sharpness: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


@dataclass
class SamplingStats:
    raw_frame_count: int = 0
    decisions: Dict[str, int] = field(default_factory=lambda: {v.value: 0 for v in SamplingVerdict})
    kept_count: int = 0

    def record(self, decision: SamplingDecision):
        self.decisions[decision.verdict.value] = self.decisions.get(decision.verdict.value, 0) + 1
        if decision.verdict in (SamplingVerdict.KEEP, SamplingVerdict.KEEP_MOTION_BURST,
                                 SamplingVerdict.KEEP_FORCED):
            self.kept_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _bbox_iou(a: Optional[Dict[str, int]], b: Optional[Dict[str, int]]) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1 = a.get("x", 0), a.get("y", 0)
    ax2, ay2 = ax1 + a.get("w", 0), ay1 + a.get("h", 0)
    bx1, by1 = b.get("x", 0), b.get("y", 0)
    bx2, by2 = bx1 + b.get("w", 0), by1 + b.get("h", 0)
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


def _optical_flow_median_magnitude(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    max_corners: int = 200,
) -> float:
    """Sparse Lucas-Kanade flow.  Returns median pixel displacement."""
    corners = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=8
    )
    if corners is None or len(corners) == 0:
        return 0.0
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)
    if next_pts is None:
        return 0.0
    valid = status.flatten() == 1
    if not np.any(valid):
        return 0.0
    deltas = (next_pts[valid] - corners[valid]).reshape(-1, 2)
    mags = np.linalg.norm(deltas, axis=1)
    return float(np.median(mags))


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class AdaptiveSampler:
    """
    Stateful sampler — caller feeds frames sequentially via `decide(...)`.

    Maintains:
        - last_kept_gray (downsampled grayscale)
        - last_kept_bbox (mask bbox)
        - raw_frame_index (counter since start)
        - frames_since_last_kept (for force-keep cadence)
    """

    def __init__(self, thresholds: Optional[SamplingThresholds] = None, downsample: int = 2):
        self.th = thresholds or SamplingThresholds()
        self.downsample = max(1, int(downsample))
        self.last_kept_gray: Optional[np.ndarray] = None
        self.last_kept_bbox: Optional[Dict[str, int]] = None
        self.raw_frame_index: int = -1
        self.frames_since_last_kept: int = 0
        self.stats = SamplingStats()

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        if self.downsample > 1:
            gray = gray[::self.downsample, ::self.downsample]
        return gray

    def decide(
        self,
        frame: np.ndarray,
        bbox: Optional[Dict[str, int]] = None,
    ) -> SamplingDecision:
        """
        Single-frame decision.  Caller should advance only on every raw input
        (not just every kept one) so the cadence counter works.
        """
        self.raw_frame_index += 1
        gray = self._to_gray(frame)

        # First frame — always keep
        if self.last_kept_gray is None:
            decision = SamplingDecision(
                verdict=SamplingVerdict.KEEP_FORCED,
                reasons=["first frame"],
                flow_median=0.0,
                bbox_iou=0.0,
                sharpness=_laplacian_variance(gray),
            )
            self._commit(decision, gray, bbox)
            return decision

        self.frames_since_last_kept += 1

        # Min-gap guard — never keep two frames back-to-back
        if self.frames_since_last_kept < self.th.min_gap_raw_frames:
            d = SamplingDecision(
                verdict=SamplingVerdict.SKIP_STATIC,
                reasons=[f"min_gap_raw_frames={self.th.min_gap_raw_frames}"],
            )
            self.stats.record(d)
            return d

        # Compute signals
        flow_median = _optical_flow_median_magnitude(self.last_kept_gray, gray)
        sharpness = _laplacian_variance(gray)
        iou = _bbox_iou(self.last_kept_bbox, bbox)

        decision = self._classify(flow_median, sharpness, iou, gray, bbox)
        self.stats.record(decision)

        if decision.verdict in (SamplingVerdict.KEEP, SamplingVerdict.KEEP_MOTION_BURST,
                                 SamplingVerdict.KEEP_FORCED):
            self._commit(decision, gray, bbox)

        return decision

    def _classify(
        self,
        flow_median: float,
        sharpness: float,
        iou: float,
        gray: np.ndarray,
        bbox: Optional[Dict[str, int]],
    ) -> SamplingDecision:
        reasons: List[str] = []
        d = SamplingDecision(
            verdict=SamplingVerdict.KEEP,
            flow_median=flow_median,
            bbox_iou=iou,
            sharpness=sharpness,
        )

        # Force-keep cadence — even on a static scene, drop one anchor periodically
        if self.frames_since_last_kept >= self.th.force_keep_every_n_raw_frames:
            d.verdict = SamplingVerdict.KEEP_FORCED
            d.reasons.append(f"force_keep_every_n={self.th.force_keep_every_n_raw_frames}")
            return d

        # Too blurry — skip
        if sharpness < self.th.min_sharpness:
            d.verdict = SamplingVerdict.SKIP_BLURRY
            d.reasons.append(f"sharpness {sharpness:.1f} < {self.th.min_sharpness}")
            return d

        # Static (no motion AND nearly identical bbox)
        if flow_median < self.th.min_flow_static and (iou == 0.0 or iou > self.th.static_iou):
            d.verdict = SamplingVerdict.SKIP_STATIC
            d.reasons.append(f"flow {flow_median:.2f} < {self.th.min_flow_static} & iou {iou:.2f} > {self.th.static_iou}")
            return d

        # Redundant — small motion + high IoU
        if flow_median < self.th.redundant_flow and iou > self.th.redundant_iou:
            d.verdict = SamplingVerdict.SKIP_REDUNDANT
            d.reasons.append(f"flow {flow_median:.2f} < {self.th.redundant_flow} & iou {iou:.2f} > {self.th.redundant_iou}")
            return d

        # Burst — extreme motion → keep but tag
        if flow_median > self.th.burst_flow:
            d.verdict = SamplingVerdict.KEEP_MOTION_BURST
            d.reasons.append(f"flow {flow_median:.2f} > burst_flow {self.th.burst_flow}")
            return d

        d.reasons.append(f"flow {flow_median:.2f}, iou {iou:.2f}, sharp {sharpness:.0f}")
        return d

    def _commit(self, decision: SamplingDecision, gray: np.ndarray, bbox: Optional[Dict[str, int]]):
        self.last_kept_gray = gray
        self.last_kept_bbox = bbox
        self.frames_since_last_kept = 0
        # Stats already recorded for non-first; record first here
        if decision.verdict == SamplingVerdict.KEEP_FORCED and self.raw_frame_index == 0:
            self.stats.record(decision)

    def reset(self):
        self.last_kept_gray = None
        self.last_kept_bbox = None
        self.raw_frame_index = -1
        self.frames_since_last_kept = 0
        self.stats = SamplingStats()
