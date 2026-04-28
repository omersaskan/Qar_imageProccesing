# Phase 6.1B: SAM2 Video-Temporal Implementation Plan
**Status:** DRAFT (Plan only)
**Target:** Robust temporal segmentation using SAM2 Video Predictor.

## 1. Overview
Instead of per-frame image inference, this phase will implement SAM2's video propagation API. This allows the model to "track" an object through time, using information from previous frames to refine the current mask.

## 2. Key Components
- **`build_sam2_video_predictor`**: Initialize the video-specific predictor from SAM2.
- **`init_state`**: Start a tracking session for a specific video/frame-sequence.
- **`add_new_points_or_box`**: Seed the tracking on the first frame (or keyframes) using:
  - Manual first-frame box (derived from GT or user input).
  - High-confidence legacy bbox.
- **`propagate_in_video`**: Run the forward (and optionally backward) propagation to generate masks for all frames.

## 3. Workflow Implementation
1. **Inference Backend**: Create `modules/ai_segmentation/sam2_video_backend.py`.
2. **State Management**:
   - Store frame sequences in a temporary directory formatted for SAM2.
   - Manage tracking IDs and state buffers.
3. **Prompt Strategy**:
   - Use `frame_0000` as the "seed" frame.
   - Provide a high-precision bounding box to `add_new_points_or_box`.
4. **Compatibility**:
   - Ensure output mask naming matches existing pipeline (`frame_0000.jpg.png`).
   - Maintain strict `fallback_to_legacy` if the video predictor fails.

## 4. Evaluation Criteria
Success will be measured against the same dev-subset using the following thresholds:
- **IoU Gain**: >= +0.05 vs Legacy.
- **Leakage Reduction**: > 0.
- **Empty Mask Count**: == 0 (Temporal consistency should prevent loss of object).
- **Jitter**: Centroid and Bbox jitter must be significantly lower than Legacy.

## 5. Security & Isolation
- Keep `SAM2_ENABLED=true` but `SAM2_MODE=video` (new flag).
- `SAM2_REVIEW_ONLY=true` remains active until validated.
- Depth Anything remains blocked.

---
**User Review Required**: Please approve this plan before implementation starts.
