# Phase 6.1: SAM2 Image-Mode Results Summary
**Date:** 2026-04-28

## 1. Tiny Sweep Results
Evaluation performed on `cap_29ab6fa1` (20 frames). Baseline Legacy IoU: **0.4502**.

| Mode | IoU | Gain | Leakage | Jitter | Empty |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **manual_first_frame_box** | **0.4656** | **+0.0155** | **0.5283** | **200.93** | 0 |
| center_point | 0.4538 | +0.0036 | 0.5416 | 203.32 | 0 |
| auto | 0.4538 | +0.0036 | 0.5416 | 203.32 | 0 |
| legacy_centroid | 0.4538 | +0.0036 | 0.5416 | 203.32 | 0 |
| center_box | 0.4531 | +0.0030 | 0.5432 | 200.89 | 0 |
| legacy_bbox | 0.4531 | +0.0030 | 0.5432 | 200.89 | 0 |

## 2. SAM2.1 Large Result
Best Tiny prompt (`manual_first_frame_box`) was tested with Large model:
- **SAM2.1 Large IoU**: 0.4562
- **IoU Gain**: +0.0060
- **Leakage Reduction**: 0.0084

## 3. Success Threshold Failure
The success threshold of **IoU Gain >= +0.05** was NOT reached by any image-mode configuration.
- Best gain: +0.0155 (Tiny, Manual Box).

## 4. Decision: Image-Mode Not Accepted
SAM2 image-mode (per-frame inference) is not accepted as the production default. The gains are marginal on this dataset, and the system is susceptible to data/GT noise.

## 5. Visual Audit (frame_0010)
A visual audit of the "failure" at frame 10 revealed a **Ground Truth error**. The GT mask labeled a background pillow, while both Legacy and SAM2 correctly segmented the foreground water bottle. This indicates that while the models are better than the metrics suggest, image-mode still lacks the temporal robustness to definitively outperform the legacy pipeline across the entire capture.

## 6. Recommendation: Evaluate Video-Temporal SAM2
The next step is to pivot to **SAM2 Video Mode** (Temporal Propagation). By leveraging memory across frames, the system can maintain consistency and reject background noise more effectively.

> [!WARNING]
> **Depth Anything remains BLOCKED**. 
> Depth prior generation is gated until a segmentation pipeline consistently achieves the required IoU and stability benchmarks.
