# Guided Capture Migration Plan

## 1. Purpose

Meshysiz Asset Factory is moving from an admin-oriented video upload workflow toward a customer-facing guided capture workflow.

The current system can accept a video, extract frames, generate masks, run reconstruction, clean the mesh, export a GLB, and validate the output. However, arbitrary video upload is not the correct long-term customer experience.

The target product experience is:

```text
Guided camera capture
  -> live capture quality scoring
  -> viewpoint coverage ring
  -> original frames + masks
  -> COLMAP + OpenMVS reconstruction
  -> cleanup / isolation / remesh
  -> texturing
  -> GLB export
  -> validation
  -> customer-ready or draft/degraded state
  -> training-data manifest
```

Capture should finish because the system has enough quality and coverage, not because a fixed timer has expired.

---

## 2. Current State

The current repository is organized around an upload-driven pipeline.

### API Upload Flow

Expected files:

- `modules/operations/api.py`
- `ui/app.js`

Current behavior:

- The UI allows selecting or dragging a video file.
- The API accepts `.mp4`, `.mov`, and `.avi` files.
- A capture session is created.
- The file is stored under the capture session directory.
- The worker later processes the session.

This is acceptable for admin, debugging, import, support, and internal test workflows. It should not remain the primary customer-facing workflow.

### Worker Pipeline

Expected file:

- `modules/operations/worker.py`

Current high-level flow:

```text
CREATED
  -> CAPTURED
  -> RECONSTRUCTED
  -> CLEANED
  -> EXPORTED
  -> VALIDATED
  -> PUBLISHED / FAILED / RECAPTURE_REQUIRED
```

The worker is responsible for stage transitions, retries, timeout handling, cleanup, export, validation, guidance, and publish.

### Frame Extraction

Expected file:

- `modules/capture_workflow/frame_extractor.py`

Current behavior:

- Opens uploaded video.
- Samples frames.
- Runs object masking.
- Rejects poor quality or redundant frames.
- Writes extracted frames and masks.
- Generates quality reports.

### Mask Generation

Expected files:

- `modules/capture_workflow/object_masker.py`
- `modules/capture_workflow/segmentation_backends/*`

Current behavior:

- Uses `rembg` as the primary segmentation backend.
- Falls back to heuristic segmentation where configured.
- Computes mask confidence, purity, occupancy, support suspicion, and bounding boxes.

### Coverage Analysis

Expected files:

- `modules/capture_workflow/coverage_analyzer.py`
- `modules/capture_workflow/geometric_analyzer.py`

Current behavior:

- Estimates readable frames.
- Estimates unique views.
- Computes coverage score.
- Detects insufficient viewpoint diversity.
- Produces recapture guidance when coverage is insufficient.

### Reconstruction Runner

Expected files:

- `modules/reconstruction_engine/runner.py`
- `modules/reconstruction_engine/adapter.py`

Current behavior:

- Runs COLMAP or COLMAP/OpenMVS depending on configuration.
- Supports fallback attempts such as denser frames.
- Selects the best reconstruction attempt.
- Writes reconstruction audit and output manifest.

### COLMAP / OpenMVS Adapter

Expected files:

- `modules/reconstruction_engine/adapter.py`
- `modules/reconstruction_engine/openmvs_texturer.py`

Current behavior:

- COLMAP is used for feature extraction, matching, sparse reconstruction, dense reconstruction, fusion, and meshing.
- OpenMVS can be used for texturing and/or reconstruction depending on configuration.
- OpenMVS binary configuration must be hardened.

### Cleanup / Remesh / Isolation

Expected files:

- `modules/asset_cleanup_pipeline/cleaner.py`
- `modules/asset_cleanup_pipeline/remesher.py`
- `modules/asset_cleanup_pipeline/isolation.py`
- `modules/asset_cleanup_pipeline/alignment.py`
- `modules/asset_cleanup_pipeline/bbox.py`

Current behavior:

- Removes likely table/support contamination.
- Splits mesh components.
- Selects the most product-like component.
- Remeshes / decimates.
- Aligns to ground.
- Generates normalized metadata.

Potential issues:

- Cleanup and remesh can remove UV/material data.
- Flat products may be mistaken for table/support.
- Category-specific cleanup profiles will be needed.

### Texturing

Expected files:

- `modules/operations/texturing_service.py`
- `modules/reconstruction_engine/openmvs_texturer.py`

Current behavior:

- Attempts OpenMVS texturing when a dense workspace exists.
- Produces textured OBJ and texture atlas when successful.
- Marks texturing as real, degraded, or absent.

### GLB Export

Expected file:

- `modules/export_pipeline/glb_exporter.py`

Current behavior:

- Loads cleaned mesh.
- Applies texture when UVs are available.
- Exports GLB.
- Inspects exported asset for UV, material, embedded texture, texture integrity, and material semantic status.

Potential issue:

- GLB generation does not mean the asset is customer-ready.
- UV-less or texture-less GLB must be treated as draft/degraded, not final.

### Validation / Publish

Expected files:

- `modules/qa_validation/*`
- `modules/operations/worker.py`
- `modules/asset_registry/*`

Required change:

- Customer-ready publish must require UV and embedded texture.
- Geometry-only GLBs must not be considered customer-ready.

---

## 3. Target State

The target architecture separates three flows.

### 3.1 Customer Guided Capture

The future customer path.

```text
Camera opens
  -> live frame quality checks
  -> object framing feedback
  -> capture ring / viewpoint sectors fill
  -> original frames + masks are collected
  -> capture completes when quality gates pass
```

### 3.2 Admin/Internal Video Upload

Kept for:

- debugging
- import
- support
- regression tests
- internal QA
- power users

This path should not be described as the final customer-facing capture experience.

### 3.3 Dataset-Ready Processing

Every session must generate structured training data:

- capture metrics
- reconstruction metrics
- export metrics
- validation result
- labels
- consent / training eligibility
- failure reason taxonomy

---

## 4. Migration Strategy

## P0 — Reliability and Data Foundations

- Create/update architecture documentation.
- Harmonize OpenMVS environment variables.
- Add OpenMVS readiness checks.
- Block simulated reconstruction in pilot/production.
- Strengthen GLB UV/texture guards.
- Ensure geometry-only GLB cannot be customer-ready.
- Add upload preflight checks.
- Make UI API base configurable.
- Add training data manifest schema, builder, and registry.
- Generate training manifest best-effort for completed, failed, and recapture sessions.

## P1 — Guided Capture Foundation

- Add guided capture API contract.
- Add original frames + masks separation.
- Add capture quality scoring module.
- Add viewpoint sector / capture ring model.
- Add product category presets.
- Add UV preservation regression tests.
- Connect guidance system to capture feedback.

## P2 — Advanced Intelligence and Scale

- Add TRELLIS/Meshy fallback router.
- Add domain-specific segmentation fine-tuning plan.
- Add dataset export and train/val/test splitting.
- Add model training scripts.
- Add human QA workflow.
- Add native mobile capture SDK or PWA capture client.

---

## 5. Non-goals For This Sprint

The following are not required in the initial migration sprint:

- Full native iOS app.
- Full native Android app.
- Full TRELLIS/Meshy integration.
- Actual model fine-tuning.
- Human QA web panel.
- Full 3D Gaussian/Splat preview system.
- Replacing COLMAP/OpenMVS as the primary engine.

---

## 6. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| UV loss during cleanup | Texture cannot be applied | UV preservation tests and post-cleanup inspection |
| Texture missing | GLB looks incomplete | Customer-ready validation blocks geometry-only output |
| OpenMVS env mismatch | Texturing fails | Canonical `settings.openmvs_path` and readiness probe |
| Weak COLMAP sparse model | Reconstruction fails | Guided capture quality and recapture flow |
| Low-texture object | Poor registration | Capture guidance and AI draft fallback |
| Reflective/transparent object | Poor geometry/texture | Category-specific guidance and fallback |
| Mask contamination | Table included in mesh | Better segmentation and mask QA metrics |
| Cleanup removes product | Product damaged | Category-specific cleanup profiles |
| Dataset without consent | Privacy risk | Eligibility and consent fields |
| Raw video retained forever | Privacy/storage risk | Retention policy and configurable cleanup |

---

## 7. Acceptance Criteria

This migration is successful when:

- The new guided capture target architecture is documented.
- Video upload is clearly marked as admin/internal.
- OpenMVS configuration is explicit and readiness-checked.
- Simulated reconstruction is blocked in pilot/production.
- GLB export honestly reports UV and texture status.
- Geometry-only assets cannot be customer-ready.
- Attempt scoring considers texture/UV quality.
- Training manifests are generated best-effort.
- Consent and eligibility fields exist.
- Raw user identity is not written into training manifests.
- Tests are added or updated for key P0 behavior.