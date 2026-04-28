# Phase 6.1: SAM2 Image-Mode Results Summary
**Date:** 2026-04-28

## 1. Evaluation Context
Evaluation performed on `cap_29ab6fa1` (20 frames). 
A visual audit revealed that **frame_0010** had a critical Ground Truth (GT) error (labeled background pillow instead of subject bottle). Corrected metrics exclude this frame.

## 2. Corrected Metrics (Excluding frame_0010)
| Model | Mode | Corrected IoU | **Corrected Gain** | Corrected Leakage |
| :--- | :--- | :--- | :--- | :--- |
| **Legacy (Rembg)** | baseline | 0.9004 | -- | 0.0952 |
| **SAM2.1 Tiny** | manual_box | **0.9313** | **+0.0309** | 0.0567 |
| **SAM2.1 Large** | manual_box | 0.9125 | +0.0121 | 0.0784 |

## 3. Raw Metrics (Including frame_0010, for traceability)
| Model | Mode | Raw IoU | Raw Gain | Raw Leakage |
| :--- | :--- | :--- | :--- | :--- |
| **Legacy (Rembg)** | baseline | 0.4502 | -- | 0.5476 |
| **SAM2.1 Tiny** | manual_box | 0.4656 | +0.0154 | 0.5283 |
| **SAM2.1 Large** | manual_box | 0.4562 | +0.0060 | 0.5392 |

## 4. Success Threshold Failure
The success threshold of **IoU Gain >= +0.05** was NOT reached by any image-mode configuration, even with corrected metrics.
- Best gain: +0.0309 (Tiny, Manual Box).

## 5. Decision: Image-Mode Not Accepted
SAM2 image-mode (per-frame inference) is not accepted as the production default. While accurate, the gain over the legacy pipeline is marginal (+3.1%) and does not justify the additional complexity and hardware requirements in its current form.

## 6. Recommendation: Evaluate Video-Temporal SAM2
Proceed to **Phase 6.1B: SAM2 Video-Temporal Mode**. By leveraging temporal consistency, we expect to achieve the target robustness and >0.05 IoU gain.

> [!WARNING]
> **Depth Anything remains BLOCKED**. 
> Depth prior generation is gated until a segmentation pipeline consistently achieves the required IoU and stability benchmarks.
