# AI 3D Phase 2C — Background Removal E2E Report

## Overview
This report documents the End-to-End (E2E) verification of Phase 2C: Optional Background Removal and Object Isolation in the AI 3D generation pipeline.

- **Commit SHA**: `ddafea7` (Base) -> `ddafea7` (with report)
- **Date**: 2026-05-04
- **Environment**: Local Windows 11 + WSL2 (Ubuntu 24.04)
- **GPU**: NVIDIA CUDA enabled

## Verification Results

### 1. Automated Tests
- **Suites**: Phase 2B, Phase 2C, Quality Profiles, Pipeline Security, Remote Provider Mocks.
- **Result**: **177 passed** ✅
- **Stability**: No regressions detected in Phase 1 multi-candidate or Phase 2A quality profiles.

### 2. Dependency Check
- **rembg installed**: **True** ✅
- **Behavior**: The preprocessor correctly identifies the presence of `rembg` and uses it when `background_removal_enabled=true`.

### 3. E2E Test Scenarios

| Scenario | Quality Mode | BG Removal | Background Removed | Mask Source | GLB Result | GLB Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E2E A** | High | OFF | **False** | `fallback_center_crop` | Success | 1.41 MB |
| **E2E B** | High | ON | **True** | `rembg` | Success | 1.46 MB |
| **E2E C** | Ultra | ON | **True** | `rembg` | Success | 1.77 MB |

### 4. Prepared Input Inspection (E2E B)
- **File**: `data/ai_3d/ai3d_a553b9fbe812/derived/ai3d_input.png`
- **Mode**: `RGBA` ✅
- **Size**: `1024x1024` ✅
- **Alpha Channel**: Present and functional (confirmed via PIL inspection).
- **Inference Stability**: SF3D (WSL subprocess) successfully consumed the RGBA PNG without errors.

### 5. Metadata Accuracy
The generated manifests correctly capture the new spatial and alpha-specific metadata:
- `background_removed`: `true`
- `mask_source`: `"rembg"`
- `bbox_source`: `"rembg_alpha"`
- `alpha_bbox`: `[96, 96, 404, 404]`
- `foreground_ratio_estimate`: `0.2842`

## Key Observations
- **Safe Fallback**: Background removal is optional and falls back safely to standard center-cropping if `rembg` is unavailable or fails to produce a valid alpha mask.
- **Provider Integrity**: **External providers were not touched** during this phase. All operations remain local/WSL.
- **Ultra Quality**: Ultra quality profiles are correctly resolving to `input_size=1024` and `texture_resolution=2048`.

## Conclusion
Phase 2C is complete. The system is stable, and the background removal feature is ready for production use.

**Phase 2D (Advanced Post-processing/Optimizations) can begin.**
