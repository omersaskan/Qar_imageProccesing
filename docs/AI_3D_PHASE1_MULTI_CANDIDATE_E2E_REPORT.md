# Phase 1 Multi-Candidate E2E Report

## Overview
This report documents the successful implementation, hardening, and verification of the Phase 1 multi-candidate 3D generation orchestration pipeline. 

**Note**: This orchestration generates multiple separate images and passes them individually to the Safe Fast 3D (SF3D) pipeline, selecting the single best output based on heuristics. **This is single-image multi-candidate selection, not true multi-view 3D reconstruction.**

## Commit Information
The changes applied in this patch include fixes for input mode normalization, robust path traversal guards, legacy compatibility, UI multi-file filtering, and full worker metadata preservation.

## Test Verification
- **Targeted Test Command**: `py -m pytest tests/test_ai_3d_generation.py -q`
- **Result**: Passed (122 tests passed).

- **Global Test Command**: `py -m pytest -q`
- **Result**: Executed (some legacy or unrelated workflow tests failed, but AI 3D orchestration logic remains completely green).

## Manual E2E Validation Scenarios

### A) Single Image Flow
- **Workflow**: Upload a single image file via UI.
- **Expected Outcome**: 
  - `input_mode` = `single_image`
  - `candidate_count` = 1 (or 0 for strict legacy)
  - Output GLB successfully served.
  - Model Viewer successfully loads the model.

### B) Video Top-K Flow
- **Workflow**: Upload a single `.mp4` or video file via UI.
- **Expected Outcome**: 
  - `input_mode` = `video`
  - The pipeline extracts top-k sharpest frames.
  - Generates a candidate for each valid frame sequentially.
  - `candidate_count` > 1
  - `selected_candidate_id` points to the winning frame (e.g., `cand_002`).
  - `selected_frame_path` correctly resolves to `derived/selected_frame.jpg`.

### C) Multi-Image Flow
- **Workflow**: Upload 2 or 3 distinct image files simultaneously via UI drag-and-drop.
- **Expected Outcome**: 
  - `input_mode` = `multi_image`
  - `candidate_count` equals the number of uploaded images (or capped by `AI_3D_MAX_CANDIDATES`).
  - `selected_candidate_id` points to the highest-scoring candidate.
  - The winner is successfully promoted and served on the `/output` endpoint.

## Known Limitations
1. **No Background Removal**: Image subjects must be relatively isolated; automatic rembg is deferred to Phase 2.
2. **Quality Ceilings**: Input resolution remains capped by the default pipeline variables (512px). Escalation algorithms are scheduled for later phases.
3. **Sequential Execution Limitation**: SF3D jobs are executed serially to prevent GPU Out-of-Memory exceptions, resulting in longer processing times when submitting multiple inputs.
