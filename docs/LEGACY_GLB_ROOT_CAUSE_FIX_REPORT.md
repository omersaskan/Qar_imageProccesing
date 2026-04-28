# LEGACY GLB ROOT CAUSE FIX REPORT

## 1. Executive Summary
The legacy photogrammetry pipeline was producing malformed, blob-like GLB assets due to a fundamental coordinate space mismatch during the asset isolation stage. By surgically correcting camera model selection and fixing mask path resolution, we have restored semantic guidance to the reconstruction process.

## 2. Root Causes Identified & Fixed

### A. Camera Model Mismatch
- **Problem**: `load_reconstruction_cameras()` was loading the original distorted **RADIAL** camera model (2160x3840) while the reconstructed mesh exists in undistorted **PINHOLE** space (1125x2000).
- **Fix**: Updated `load_reconstruction_cameras` to prioritize the undistorted model in `dense/sparse/`.
- **Impact**: 3D-to-2D projection for mask-support scoring is now geometrically accurate.

### B. Mask Path Resolution Bug
- **Problem**: Mask loading was failing when the workspace path was the `dense/` directory, causing a fallback to weak geometric-only isolation.
- **Fix**: Expanded search paths in `load_reconstruction_masks` to include `stereo/masks` and added automatic nearest-neighbor resizing if mask dimensions differ from the camera model.
- **Impact**: Semantic guidance is now consistently loaded and correctly scaled.

### C. Semantic Quality Gate & Validation Refinement
- **Problem**: The pipeline would export assets even if isolation failed semantically. Furthermore, validation was incorrectly penalizing small clean objects for being a minor part of a large original scene.
- **Fix**: 
  - Added a post-cleanup quality gate in `cleaner.py` (reject if `primary_face_share` < 50%).
  - Refined `rules.py` to use `largest_kept_component_share` (dominance within output) as the primary gate.
  - Decoupled `kept_to_initial_face_ratio` as an informational metric.
- **Impact**: Prevents contaminated assets from reaching delivery while allowing small, clean isolates to pass validation.

## 3. Files Changed
- `modules/asset_cleanup_pipeline/camera_projection.py`: Corrected loader logic and added auto-resize.
- `modules/asset_cleanup_pipeline/isolation.py`: Added detailed quality metrics.
- `modules/asset_cleanup_pipeline/cleaner.py`: Implemented the hard quality gate.
- `modules/operations/worker.py`: Normalized workspace paths.
- `modules/reconstruction_engine/adapter.py`: Clarified dense mask metrics.
- `modules/qa_validation/rules.py`: Updated validation logic to check new quality flags.
- `modules/export_pipeline/glb_exporter.py`: Clarified structural vs semantic readiness.

## 4. Evidence of Effectiveness

### A. Regression Testing
- Verified `load_reconstruction_cameras` preference for `dense/sparse` (undistorted space).
- Verified `load_reconstruction_masks` directory discovery and auto-resizing.
- Verified semantic quality gate logic via mock mesh components.

### B. Standardized Coordinate Space
By enforcing the use of the `dense/` workspace for cleanup guidance, we eliminate the previous jitter and mis-projections where masks were being applied to the wrong camera model (RADIAL vs PINHOLE).

### C. Forced Quality Gates
Assets with < 50% primary component share are now automatically rejected, preventing the delivery of "blob-like" contamination.

## 5. Verification Status
- **Regression Tests**: Passed.
- **E2E Job `legacy_cap_29ab6fa1_compare_v3`**: 
  - **Status**: Running (`patch_match_stereo`).
  - **Empirical Evidence**: 
    - `Source mask dimension matches (pre-resize): 0` (Confirmed mismatch detected).
    - `Dense mask dimension matches (post-write): 54` (Confirmed auto-resize successful).
    - `Resized masks: 54` (Confirmed fix applied to all frames).
  - **Selected Camera Model**: PINHOLE (Undistorted dense).

---
*Status: Verification in Progress (Reconstruction Step).*
