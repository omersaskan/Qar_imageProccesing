# AI 3D Phase 2E — Preprocessing-Aware Scoring E2E Report

## Overview
This report documents the End-to-End (E2E) validation of Phase 2E: Preprocessing-aware candidate scoring in the AI 3D generation pipeline.

- **Commit SHA**: `2e79518` (Base) -> `2e79518` (Report Addition)
- **Date**: 2026-05-05
- **Environment**: Local Windows 11 + WSL2 (Ubuntu 24.04)
- **GPU**: NVIDIA CUDA enabled

## Verification Results

### 1. Targeted Tests
- **Suites**: Phase 2D Scoring, Phase 2C Background Removal, Phase 2B Preprocessing, Quality Profiles, Pipeline Security.
- **Result**: **182 passed** ✅
- **Stability**: No regressions detected in core generation or security gates.

### 2. Preflight Summary
- **Execution Mode**: `wsl_subprocess` ✅
- **WSL Checks**: Passed (Python, Distro, Worker Script) ✅
- **rembg installed**: **True** ✅

### 3. E2E Test Scenarios

#### E2E A: Multi-image Scoring (Background Removal ON)
- **Session**: `ai3d_8ab48e8f2cab`
- **Inputs**: `clean_object.png` (Centered), `poor_object.png` (Small/Off-center)
- **Result**: **Success** ✅
- **Observation**: `cand_001` (Clean) scored significantly higher than `cand_002` (Poor).
- **Selected**: `cand_001` (Score: 114.93)
- **Scoring Breakdowns**:
  - `cand_001`: `background_removed_bonus: 8.0`, `rembg_bonus: 9.0`, `foreground_ratio_score: 6.0`.
  - `cand_002`: `foreground_ratio_score: -8.0` (due to small size).

#### E2E B: Multi-image Scoring (Background Removal OFF)
- **Session**: `ai3d_86d209d11678`
- **Result**: **Success** ✅
- **Observation**: Background removal correctly bypassed.
- **Penalties**: `fallback_penalty: -5.0` (via `fallback_center_crop`) correctly applied.
- **Scoring Breakdown**: `background_removed_bonus: 0.0`.

#### E2E C: Video Candidate Scoring (Background Removal ON)
- **Session**: `ai3d_bfb0d453189e`
- **Input**: `dummy_video.mp4` (Synthetic 3nd loop)
- **Result**: **Success** ✅
- **Observation**: 3 frames extracted and scored.
- **Selected**: `cand_003` (Score: 114.94)
- **Manifest**: `input_mode: video`, `candidate_ranking` contains compact preprocessing fields.

### 4. Candidate Ranking Excerpt (E2E C)
```json
"candidate_summary": [
  {
    "candidate_id": "cand_003",
    "score": 114.94,
    "status": "ok",
    "selected": true,
    "background_removed": true,
    "mask_source": "rembg",
    "foreground_ratio_estimate": 0.3065
  },
  ...
]
```

### 5. Scoring Example (Breakdown Audit)
From `cand_001` (E2E A):
- `background_removed_bonus`: 8.0
- `rembg_bonus`: 9.0
- `foreground_ratio_score`: 6.0
- `fallback_penalty`: 0.0
- `bbox_sanity_score`: 3.0
- **Final Score**: 114.93

## Conclusion
Phase 2D scoring is validated in real E2E candidate selection. The pipeline correctly prioritizes geometric-quality inputs and penalizes fallbacks or poor segmentation results.

**Phase 3 Benchmark (External Providers) can begin.**

**External providers were not touched.**
