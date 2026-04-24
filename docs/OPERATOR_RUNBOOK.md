# Meshysiz Asset Factory — Operator Runbook

This document explains how to deploy, operate, monitor, and troubleshoot **Meshysiz Asset Factory** in local, pilot, and production environments.

This runbook covers:

- environment configuration
- API authentication
- readiness checks
- COLMAP and OpenMVS requirements
- worker monitoring
- guided capture principles
- GLB customer-ready rules
- texture and UV validation
- AI fallback policy
- training-data manifest generation
- retention and privacy
- common failure cases
- deployment and operator approval checklists

This runbook no longer describes only the legacy:

```text
video upload -> reconstruction -> GLB
```

flow.

It now covers the target architecture:

- **Guided Capture / AR-like capture**
- **Reliable COLMAP + OpenMVS reconstruction**
- **Texture/UV-safe GLB export**
- **Customer-ready GLB validation**
- **AI fallback / draft separation**
- **Training-data-ready capture logging**

---

## 1. System Overview

Meshysiz Asset Factory transforms product captures into 3D assets such as GLB/USDZ.

The long-term customer-facing flow is:

```text
Customer Guided Capture
  -> Capture quality scoring
  -> Original frames + masks
  -> COLMAP + OpenMVS reconstruction
  -> Cleanup / isolation / remesh
  -> Texturing
  -> GLB export
  -> Validation
  -> Customer-ready publish or draft/degraded state
  -> Training manifest generation
```

The current admin/internal flow is:

```text
Admin video upload
  -> Frame extraction
  -> Mask generation
  -> Coverage analysis
  -> Reconstruction
  -> Cleanup
  -> Export
  -> Validation
  -> Publish / fail / recapture
  -> Training manifest generation
```

> Important: `POST /api/sessions/upload` is **not** the future customer-facing capture path. It is retained for admin, debug, import, support, internal test, and regression workflows. The customer-facing target is guided camera / AR-like capture.

---

## 2. Environment Profiles

The system recognizes three profiles through the `ENV` environment variable.

| ENV | Purpose | Security | Logging | Stubs |
|---|---|---|---|---|
| `local_dev` | Development | Relaxed | Verbose / debug-friendly | Explicit permission only |
| `pilot` | Controlled pilot/customer testing | API key required | Structured JSON / INFO | Forbidden |
| `production` | Live production | API key required | Structured JSON / WARNING | Forbidden |

### 2.1 Required Environment Variables

Example `.env`:

```bash
# Core
ENV=pilot
DATA_ROOT=data
PILOT_API_KEY=your_secure_random_key_here

# Reconstruction binaries
RECON_ENGINE_PATH=C:\colmap\colmap.exe
OPENMVS_BIN_PATH=C:\openmvs\bin

# Backward compatibility only.
# If both are present, OPENMVS_BIN_PATH should be treated as canonical.
OPENMVS_BIN=C:\openmvs\bin

# Reconstruction engine
RECON_PIPELINE=colmap_openmvs
RECON_USE_GPU=true

# Texture policy
REQUIRE_TEXTURED_OUTPUT=true
OPENMVS_TEXTURED_OUTPUT=true

# Fallbacks
RECON_FALLBACK_STEPS=default,denser_frames
RECON_UNMASKED_FALLBACK_ENABLED=false
RECON_FALLBACK_SAMPLE_RATE=5

# Worker
WORKER_INTERVAL_SEC=5

# Retention
RETENTION_PUBLISHED_FRAMES_DAYS=3
RETENTION_FAILED_FRAMES_DAYS=14
RETENTION_DRAFT_FRAMES_DAYS=7
RETENTION_RECON_SCRATCH_HOURS=48

# Upload preflight
MAX_UPLOAD_MB=500
MIN_VIDEO_DURATION_SEC=8
MAX_VIDEO_DURATION_SEC=120
MIN_VIDEO_WIDTH=720
MIN_VIDEO_HEIGHT=720
MIN_VIDEO_FPS=20
MIN_FREE_DISK_GB=5

# Training data
TRAINING_DATA_ENABLED=true
TRAINING_DEFAULT_CONSENT_STATUS=unknown
```

---

## 3. Engine Configuration Policy

### 3.1 Primary Engine

The primary reconstruction engine is:

```text
COLMAP + OpenMVS
```

Preferred pipeline for pilot and production:

```bash
RECON_PIPELINE=colmap_openmvs
```

### 3.2 Supported Pipeline Values

| Value | Meaning | Pilot/Prod |
|---|---|---|
| `colmap_openmvs` | COLMAP sparse/dense + OpenMVS mesh/texture | Preferred |
| `colmap_dense` | COLMAP-only dense reconstruction | Fallback/internal |
| `simulated` | Stub/demo reconstruction | Forbidden |

### 3.3 Simulated Reconstruction Guard

`simulated` or stub reconstruction is strictly forbidden in `pilot` and `production`.

This configuration must produce a readiness issue and block ingestion:

```bash
RECON_PIPELINE=simulated
```

### 3.4 Texture Requirement

Customer-facing assets should require texture.

Recommended pilot/production value:

```bash
REQUIRE_TEXTURED_OUTPUT=true
```

If the system generates a GLB but the asset has no UV or embedded texture, the asset must **not** be considered customer-ready.

It may only be stored as one of the following states:

```text
draft_asset
geometry_only
texture_degraded
uv_only
```

---

## 4. API Authentication

In `pilot` and `production`, protected endpoints require the `X-API-KEY` header.

| Header | Description | Required In |
|---|---|---|
| `X-API-KEY` | Must match `PILOT_API_KEY` from `.env` | Pilot, Production |

### 4.1 Public Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness check. Does not require auth. |

### 4.2 Protected Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/ready` | Environment, binary, dependency, disk, and policy checks |
| `POST /api/sessions/upload` | Admin/internal video upload |
| `GET /api/worker/status` | Worker runtime status |
| `GET /api/sessions/{id}/guidance` | Session guidance |
| `GET /api/sessions/{id}/guidance/summary` | Human-readable guidance |
| `GET /api/products` | Product/session listing |
| `GET /api/products/{product_id}/history` | Product version/session history |
| `GET /api/logs` | Operational logs |
| `GET /api/training/manifests` | Internal training manifest list |
| `GET /api/training/manifests/{session_id}` | Internal training manifest detail |

---

## 5. Health and Readiness

### 5.1 Liveness

```http
GET /api/health
```

Expected response:

```json
{
  "status": "ok",
  "env": "pilot"
}
```

### 5.2 Readiness

```http
GET /api/ready
```

Readiness may return HTTP 200 while still reporting:

```json
{
  "status": "not_ready"
}
```

Operators must always inspect the `issues` field.

Expected ready response:

```json
{
  "status": "ready",
  "env": "pilot",
  "issues": [],
  "dependencies": {
    "ml_segmentation_ready": true,
    "critical_processing_ready": true
  },
  "preflight": {
    "colmap_probe_ok": true,
    "openmvs_probe_ok": true,
    "disk_ok": true,
    "simulated_pipeline_blocked": false
  }
}
```

### 5.3 Readiness Must Check

`/api/ready` must check at least the following.

#### Core

- `DATA_ROOT` exists or can be created
- free disk >= `MIN_FREE_DISK_GB`
- `ENV` is valid
- `PILOT_API_KEY` is present in `pilot` / `production`

#### COLMAP

- `RECON_ENGINE_PATH` exists
- COLMAP executable responds to `--help` or an equivalent probe
- COLMAP binary is not merely present but executable

#### OpenMVS

Canonical OpenMVS path source:

```text
settings.openmvs_path
```

Backward-compatible env vars:

```text
OPENMVS_BIN_PATH
OPENMVS_BIN
```

Required OpenMVS binaries:

| Binary | Required For |
|---|---|
| `InterfaceCOLMAP` / `InterfaceCOLMAP.exe` | COLMAP -> OpenMVS scene conversion |
| `DensifyPointCloud` / `.exe` | Dense point cloud |
| `ReconstructMesh` / `.exe` | Mesh reconstruction |
| `TextureMesh` / `.exe` | Textured OBJ/mesh output |

#### Python Dependencies

- `rembg`
- `onnxruntime` or `onnxruntime-gpu`
- `fast_simplification`
- `trimesh`
- `opencv-python`
- `numpy`
- `pydantic`
- `pydantic-settings`

#### Policy Checks

- `RECON_PIPELINE=simulated` is forbidden in pilot/prod
- customer-ready mode should require textured output
- upload should be disabled if critical dependencies are missing in pilot/prod

---

## 6. Python Dependency Verification

### 6.1 Required Packages

| Package | Purpose |
|---|---|
| `rembg` | ML-first object segmentation |
| `onnxruntime` | CPU inference backend |
| `onnxruntime-gpu` | Optional GPU inference backend |
| `fast-simplification` | Mesh decimation |
| `trimesh` | Mesh loading/export/GLB inspection |
| `opencv-python` | Video/frame analysis |
| `numpy` | Numeric operations |
| `pydantic` / `pydantic-settings` | Config/contracts |

### 6.2 CPU Installation

```powershell
pip install rembg onnxruntime fast-simplification trimesh opencv-python numpy pydantic pydantic-settings
```

### 6.3 Optional GPU Inference

```powershell
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### 6.4 Notes

- GPU inference is optional.
- Missing ML dependencies should fail readiness in pilot/prod.
- Missing ML dependencies may be warning-level only in `local_dev`.

---

## 7. Capture Modes

### 7.1 Customer Guided Capture

Long-term customer-facing capture mode.

Expected behavior:

```text
Camera opens
  -> Live capture quality metrics computed
  -> Capture ring/levels fill by viewpoint coverage
  -> User gets live feedback
  -> Capture finishes when quality threshold is met
```

Capture should not end only because a fixed time limit was reached.

### 7.2 Admin/Internal Video Upload

Endpoint:

```text
POST /api/sessions/upload
```

This is retained for:

- admin testing
- debugging
- import of already captured videos
- support workflows
- power user/manual upload
- regression testing

This endpoint should not be described as the final customer capture UX.

### 7.3 Capture Quality Principle

A capture is accepted because:

```text
quality_score >= threshold
minimum gates are satisfied
coverage is sufficient
```

not because:

```text
video duration reached 90 seconds
```

---

## 8. Capture Quality Gates

Guided capture and upload preflight should evaluate the following metrics when possible.

### 8.1 Live / Derived Metrics

| Metric | Purpose |
|---|---|
| blur score | Reject blurry frames |
| exposure score | Reject under/overexposed frames |
| object occupancy | Ensure object is visible and not too small/large |
| object clipped | Detect object cut off by frame boundaries |
| mask confidence | Estimate segmentation reliability |
| mask purity | Detect contamination/table/support |
| support/table suspicion | Detect table included in object mask |
| motion speed | Detect too-fast movement |
| feature richness proxy | Estimate COLMAP feature quality |
| optical flow/parallax proxy | Estimate viewpoint movement |
| unique view sector count | Ensure orbit coverage |
| horizontal orbit coverage | Ensure enough side views |
| elevated/top orbit coverage | Ensure top-angle evidence |
| center stability | Ensure object remains framed |
| scale variation | Detect depth/orbit variation |
| aspect variation | Detect viewpoint variation |

### 8.2 Suggested Score Bands

| Capture Score | Meaning | Action |
|---:|---|---|
| 0-39 | Reject | Recapture |
| 40-59 | Weak | Recapture recommended |
| 60-74 | Acceptable draft | Allow draft/internal |
| 75-89 | Production candidate | Proceed |
| 90-100 | Strong capture | Proceed confidently |

### 8.3 Minimum Gates

Capture should not be considered complete unless these gates are satisfied:

- minimum readable frames
- minimum unique views
- minimum blur pass ratio
- maximum clipped-frame ratio
- minimum mask confidence
- minimum object occupancy
- elevated/top view evidence for relevant categories
- no severe table/support contamination

---

## 9. Frame and Mask Storage Policy

### 9.1 Golden Rule

Do not rely only on black-background masked frames.

For dataset and reconstruction reliability, keep separate:

```text
original_frames/
masked_frames/
masks/
thumbnails/
reports/
```

### 9.2 Why

- COLMAP may need original visual context and stable features.
- Masks should be passed as masks, not only baked into black-background images.
- Training/fine-tuning later needs original + mask + outcome labels.
- Debugging segmentation failure requires original frame evidence.

### 9.3 Suggested Capture Directory Layout

```text
data/captures/{session_id}/
  video/
    raw_video.mp4
  original_frames/
  masked_frames/
  frames/
  masks/
  thumbnails/
  reports/
    capture_report.json
    quality_report.json
    coverage_report.json
    reconstruction_audit.json
    cleanup_stats.json
    export_metrics.json
    validation_report.json
    training_manifest.json
```

---

## 10. Reconstruction Pipeline

### 10.1 Main Pipeline

Recommended production path:

```text
COLMAP sparse
  -> COLMAP dense / workspace
  -> OpenMVS conversion
  -> OpenMVS densify
  -> OpenMVS reconstruct mesh
  -> OpenMVS texture mesh
  -> Cleanup / isolation
  -> UV/texture-aware GLB export
  -> Validation
```

### 10.2 Fallbacks

| Fallback | Allowed When |
|---|---|
| `denser_frames` | Default attempt too weak |
| `unmasked` | Explicitly enabled and mask likely harmful |
| `colmap_dense` fallback | Only when textured output is not strictly required |
| AI draft fallback | Future / non-critical / quick preview |
| recapture | Capture quality or reconstruction too weak |

### 10.3 COLMAP-only Output Policy

COLMAP-only output may be useful for debugging or internal draft.

It should not be assumed customer-ready unless final GLB validation passes texture/UV requirements.

---

## 11. GLB Export and Customer-Ready Policy

### 11.1 Important Distinction

A GLB can exist and still fail customer readiness.

```text
GLB generated != customer-ready asset
```

### 11.2 Customer-Ready Requirements

A customer-ready GLB should satisfy:

- `has_uv = true`
- `has_material = true`
- `has_embedded_texture = true`
- `texture_integrity_status = complete` or acceptable configured level
- `material_semantic_status >= diffuse_textured`
- `component_count` sane
- bbox dimensions sane
- no severe table/support contamination
- validation decision pass

### 11.3 Non-Customer-Ready States

| State | Meaning |
|---|---|
| `geometry_only` | Mesh exists but no usable UV/texture |
| `uv_only` | UV exists but no embedded texture |
| `texture_degraded` | Texture pipeline partially failed |
| `draft_asset` | Internal/draft only |
| `recapture_required` | Capture should be repeated |
| `reconstruction_failed` | Engine failed or too weak |
| `customer_ready` | Passed final quality gates |

### 11.4 Texture/UV Rules

- If texture path exists but mesh has no UV, texture cannot be correctly applied.
- If UV exists and texture exists but material slot is unsupported, exporter should use PBR TextureVisuals fallback.
- Exported GLB must be reloaded and inspected.
- `texture_applied_successfully` should only be trusted if exported GLB inspection confirms embedded texture.

---

## 12. Validation and Publish Policy

### 12.1 Publish Blocking

The system must not publish geometry-only or UV-less GLB as customer-ready.

Validation must distinguish:

```text
geometry_only
uv_only
diffuse_textured
pbr_partial
pbr_complete
```

### 12.2 Draft vs Published

| Condition | Publish State |
|---|---|
| Texture + UV + validation pass | `published` / `customer_ready` |
| Mesh exists but no texture | `draft` / `geometry_only` |
| UV exists but texture missing | `draft` / `uv_only` |
| Texture degraded | `draft` / `texture_degraded` |
| Weak capture | `needs_recapture` |
| Engine/config failure | `failed` |

### 12.3 Operator Rule

Operators should not approve assets based only on the presence of a `.glb` file.

They must review validation metrics.

---

## 13. AI Fallback Policy

### 13.1 Purpose

TRELLIS, Meshy, or other image-to-3D/generative engines are fallback/draft systems, not the primary trusted scan engine.

### 13.2 Use AI Fallback When

- low texture object
- weak COLMAP registration
- OpenMVS texture failure
- user wants quick preview
- internal demo
- non-critical asset
- draft asset acceptable

### 13.3 Do Not Use AI Fallback When

- exact product fidelity is required
- branded packaging/logos must be accurate
- dimensions must be reliable
- menu item must match real geometry
- user expects a faithful scan

### 13.4 Metadata Requirement

AI-generated asset metadata must explicitly include:

```json
{
  "source_engine": "trellis_or_meshy",
  "fidelity_type": "generated_estimate",
  "not_exact_scan": true,
  "needs_human_review": true
}
```

Photogrammetry asset metadata should include:

```json
{
  "source_engine": "colmap_openmvs",
  "fidelity_type": "reconstructed_scan",
  "texture_status": "complete",
  "uv_status": "present",
  "capture_score": 84.2
}
```

---

## 14. Training Data Policy

### 14.1 Principle

Every capture session should be dataset-ready.

This does not mean every capture is eligible for training. It means each session should produce structured metadata that can later become ML/fine-tuning data if consent and retention rules allow it.

### 14.2 Future ML Use Cases

Training data may support:

- capture success predictor
- failure reason classifier
- capture guidance recommender
- segmentation fine-tuning
- product category preset optimizer
- reconstruction parameter selection
- AI fallback router scoring

### 14.3 Training Manifest

Each session should produce:

```text
data/training_manifests/{session_id}.json
data/captures/{session_id}/reports/training_manifest.json
```

### 14.4 Training Registry

Global index:

```text
data/training_registry/index.jsonl
```

Each JSONL row should include:

```json
{
  "session_id": "cap_xxx",
  "created_at": "...",
  "capture_mode": "admin_video_upload",
  "product_category": "unknown",
  "capture_score": 82.5,
  "customer_ready": true,
  "failure_reason": null,
  "manifest_path": "data/training_manifests/cap_xxx.json",
  "eligible_for_training": false,
  "consent_status": "unknown"
}
```

### 14.5 Consent and Eligibility

Training eligibility is not automatic.

| Consent Status | eligible_for_training |
|---|---|
| `unknown` | false |
| `denied` | false |
| `granted` | true |
| `internal_only` | false or internal-only depending on policy |

### 14.6 Privacy Rules

Training manifest must not include raw user identity.

Required:

- hash `product_id`
- do not include user identifiers
- strip or ignore EXIF/location metadata
- separate billing/user identity from training manifest
- honor deletion requests
- support revoking training eligibility

---

## 15. Data Retention Policy

The `IngestionWorker` runs periodic cleanup. Retention should be configurable by environment.

| Artifact Type | Default Retention | Notes |
|---|---:|---|
| Raw video | Short retention, e.g. 7-30 days | Do not retain forever by default |
| Published frames | 3 days | Unless training consent allows longer |
| Failed/review frames | 14 days | Debug-oriented |
| Draft session frames | 7 days | Configurable |
| Reconstruction scratch | 48 hours | Heavy intermediate data |
| Final GLB | While product active | Permanent relative to product lifecycle |
| Manifests/reports | Long-term | Useful for audit and ML |
| Training manifests | Versioned/permanent | No raw identity |
| Training registry index | Versioned/permanent | Consent-aware |
| Audit history | Permanent | Operational history |

### 15.1 Important

Raw video should not be retained indefinitely by default.

Training-approved selected frames, masks, reports, and metrics may be retained longer only under consent and privacy policy.

---

## 16. Observability

### 16.1 Logs

Logs are stored in:

```text
data/logs/factory.log
```

Pilot/prod logs should be structured JSON.

Recommended fields:

| Field | Description |
|---|---|
| `timestamp` | UTC ISO8601 |
| `level` | INFO/WARNING/ERROR |
| `component` | api/worker/extractor/reconstruction/exporter/etc. |
| `stage` | extraction/reconstruction/cleanup/export/validation |
| `session_id` | Capture session ID |
| `job_id` | Reconstruction job ID |
| `duration_ms` | Stage duration |
| `env` | local_dev/pilot/production |
| `message` | Human-readable message |

### 16.2 Important Metrics

Operators should monitor:

- sessions created per hour
- frame extraction failure rate
- recapture_required rate
- reconstruction failure rate
- texture_degraded rate
- geometry_only rate
- customer_ready rate
- average capture_score
- average registered_images
- OpenMVS failure rate
- GLB validation failure reasons
- training manifest generation failures
- disk free space

---

## 17. Worker Operation

### 17.1 Worker Status

```http
GET /api/worker/status
```

Expected:

```json
{
  "embedded": true,
  "running": true
}
```

### 17.2 Worker Lock

The worker may use a lock file:

```text
data/worker.process
```

If a session appears stuck, check:

- worker running
- lock file stale
- factory logs
- retry count
- last pipeline progress timestamp
- environment readiness

### 17.3 Session Status Flow

Expected high-level flow:

```text
CREATED
  -> CAPTURED
  -> RECONSTRUCTED
  -> CLEANED
  -> EXPORTED
  -> VALIDATED
  -> PUBLISHED / FAILED / RECAPTURE_REQUIRED / DRAFT
```

### 17.4 Training Manifest Generation

Training manifest should be generated best-effort after:

- validation
- publish
- failure
- recapture_required

Manifest generation failure must not break asset publish. It should log a warning.

---

## 18. Upload Preflight Policy

`POST /api/sessions/upload` is admin/internal and should reject clearly invalid videos before ingestion.

Recommended checks:

| Check | Purpose |
|---|---|
| extension | Allow `.mp4`, `.mov`, `.avi` |
| file size | Prevent huge uploads |
| duration | Reject too short/long |
| fps | Reject unusably low FPS |
| resolution | Reject too low resolution |
| readability | Ensure OpenCV can open file |
| orientation | Best-effort metadata |
| codec | Best-effort warning/fail |

Recommended failure response should be explicit and actionable.

Example:

```json
{
  "detail": "Video resolution too low: 480x360. Minimum required: 720x720."
}
```

---

## 19. Common Troubleshooting

### 19.1 `/api/ready` returns `not_ready`

Possible causes:

- `RECON_ENGINE_PATH` wrong
- COLMAP binary exists but not executable
- `OPENMVS_BIN_PATH` wrong
- OpenMVS binaries missing
- Python dependencies missing
- disk space too low
- `PILOT_API_KEY` missing in pilot/prod
- `RECON_PIPELINE=simulated` in pilot/prod

Action:

1. Check `/api/ready` response `issues`.
2. Verify `.env`.
3. Verify binary paths.
4. Run dependency install.
5. Restart service.

---

### 19.2 Upload returns `503 Service Unavailable`

Cause:

```text
System Environment Incomplete
```

Usually missing dependencies or readiness failure in pilot/prod.

Action:

- Check `/api/ready`
- Install missing packages
- Verify COLMAP/OpenMVS
- Restart API/worker

---

### 19.3 Upload returns `401 Unauthorized`

Cause:

- Missing `X-API-KEY`
- Wrong API key
- `.env` key mismatch

Action:

- Verify request header
- Verify `PILOT_API_KEY`
- Restart service if env changed

---

### 19.4 Session stuck in `CREATED`

Possible causes:

- Worker not running
- Worker crashed
- lock file stale
- frame extraction repeatedly failing
- upload file unreadable
- retry limit reached

Action:

1. Check `GET /api/worker/status`.
2. Check `data/logs/factory.log`.
3. Check `data/worker.process`.
4. Check session JSON retry count.
5. Check whether video exists under `data/captures/{session_id}/video/raw_video.mp4`.

---

### 19.5 Frame extraction produced 0 frames

Possible causes:

- unreadable video
- too strict blur/exposure/mask thresholds
- segmentation failure
- object too small/large
- object clipped
- video too short
- motion too fast

Action:

- Inspect `quality_report.json`
- Inspect `coverage_report.json`
- Try better lighting / slower capture
- Use guided capture instead of arbitrary video upload
- Confirm rembg/onnxruntime installed

---

### 19.6 Capture marked `RECAPTURE_REQUIRED`

Possible causes:

- too few readable frames
- insufficient unique views
- low coverage score
- missing elevated/top view
- low mask confidence
- too many fallback masks
- poor parallax

Action:

- Ask user/operator to recapture
- Ensure product is centered
- Use slower orbit
- Add elevated/top-angle pass
- Improve lighting
- Keep object fixed and move camera

---

### 19.7 Reconstruction failed

Possible causes:

- COLMAP binary not configured
- weak sparse reconstruction
- too few registered images
- too few sparse points
- dense fusion failed
- masks too aggressive
- object low texture / reflective / transparent

Action:

- Inspect `reconstruction.log`
- Inspect `reconstruction_audit.json`
- Check registered image count
- Try denser frame fallback
- If masks are too aggressive, test controlled unmasked fallback
- Recapture if sparse registration is weak

---

### 19.8 OpenMVS texturing failed

Possible causes:

- `OPENMVS_BIN_PATH` wrong
- `InterfaceCOLMAP` missing
- `TextureMesh` missing
- COLMAP dense workspace missing
- selected mesh path invalid
- OpenMVS command failure
- image folder path mismatch

Action:

- Check `/api/ready` OpenMVS section
- Inspect `texturing.log`
- Verify `OPENMVS_BIN_PATH`
- Verify `dense/` workspace still exists
- Verify selected mesh exists
- Ensure reconstruction scratch not pruned

---

### 19.9 GLB generated but not customer-ready

Possible causes:

- no UV
- no embedded texture
- material incomplete
- texture atlas not embedded
- geometry-only output
- cleanup lost UV/material
- OpenMVS texturing degraded

Action:

- Inspect `export_metrics.json`
- Check:
  - `has_uv`
  - `has_material`
  - `has_embedded_texture`
  - `texture_integrity_status`
  - `material_semantic_status`
- Treat as draft/degraded.
- Do not approve as customer-ready.
- Recapture or rerun texture pipeline.

---

### 19.10 UV missing after cleanup

Possible causes:

- remesh destroyed UV
- isolated component lost visual/material data
- OBJ export/import stripped UV
- topology changed after texturing

Action:

- Inspect cleanup stats:
  - `uv_preserved`
  - `material_preserved`
- Prefer texture after final cleanup mesh.
- Add UV preservation regression test.
- Treat output as draft if UV missing.

---

### 19.11 Table or support included in mesh

Possible causes:

- segmentation mask included table
- support removal failed
- horizontal plane removal not enough
- object and table visually merged
- low contrast background

Action:

- Inspect masks and debug masks
- Inspect isolation stats:
  - removed plane faces
  - support bands
  - component count
- Recapture with contrasting background
- Use product category preset
- Improve segmentation model/preset

---

### 19.12 Product geometry removed during cleanup

Possible causes:

- product is flat/horizontal and mistaken for table/support
- aggressive support-band removal
- wrong category preset
- component scorer selected wrong component

Action:

- Inspect pre-cleanup mesh
- Inspect `isolation` stats
- Compare raw mesh vs cleaned mesh
- Lower plane/support removal aggressiveness for flat products
- Add category-specific cleanup profile

---

### 19.13 Training manifest not generated

Possible causes:

- manifest builder error
- missing session report path
- training data module disabled
- permissions/write failure
- invalid JSON in source reports

Action:

- Check worker warnings
- Check:
  - `data/training_manifests/`
  - `data/captures/{session_id}/reports/training_manifest.json`
  - `data/training_registry/index.jsonl`
- Manifest generation should not block publish, but should be fixed for ML readiness.

---

### 19.14 Simulated reconstruction blocked in pilot/prod

Cause:

```bash
RECON_PIPELINE=simulated
```

Action:

Set:

```bash
RECON_PIPELINE=colmap_openmvs
```

Restart service and verify `/api/ready`.

---

## 20. Operator Approval Checklist

Before marking an asset customer-ready, verify:

- [ ] Validation report exists
- [ ] Export metrics exist
- [ ] `has_uv = true`
- [ ] `has_embedded_texture = true`
- [ ] `material_semantic_status` is at least `diffuse_textured`
- [ ] `texture_integrity_status` acceptable
- [ ] Component count sane
- [ ] BBox sane
- [ ] No severe table/support contamination
- [ ] Final GLB opens in viewer
- [ ] Training manifest generated
- [ ] Consent/eligibility fields present
- [ ] Asset not AI-generated estimate unless explicitly allowed

---

## 21. Deployment Checklist

### 21.1 Before Start

- [ ] `.env` exists
- [ ] `ENV` correct
- [ ] `PILOT_API_KEY` set in pilot/prod
- [ ] `DATA_ROOT` writable
- [ ] COLMAP installed
- [ ] OpenMVS installed
- [ ] Python dependencies installed
- [ ] Free disk sufficient
- [ ] `RECON_PIPELINE=colmap_openmvs` in pilot/prod
- [ ] `REQUIRE_TEXTURED_OUTPUT=true` for customer-facing pilot/prod
- [ ] `RECON_PIPELINE` is not `simulated`
- [ ] API starts
- [ ] Worker starts
- [ ] `/api/health` ok
- [ ] `/api/ready` ready

### 21.2 After First Job

- [ ] Session status progressed
- [ ] Frames extracted
- [ ] Quality report generated
- [ ] Coverage report generated
- [ ] Reconstruction manifest generated
- [ ] Cleanup stats generated
- [ ] Export metrics generated
- [ ] Validation report generated
- [ ] GLB opens
- [ ] Texture/UV metrics pass
- [ ] Training manifest generated
- [ ] Training registry updated

---

## 22. Appendix: Recommended Directory Structure

```text
data/
  captures/
    cap_xxxxxxxx/
      video/
      original_frames/
      masked_frames/
      frames/
      masks/
      thumbnails/
      reports/
  reconstructions/
    job_cap_xxxxxxxx/
  cleaned/
    job_cap_xxxxxxxx/
  registry/
    blobs/
    meta/
  training_manifests/
    cap_xxxxxxxx.json
  training_registry/
    index.jsonl
  logs/
    factory.log
```

---

## 23. Appendix: Key Operational Concepts

### Customer-ready

A final asset that passed texture, UV, geometry, contamination, validation, and publish checks.

### Draft asset

A generated asset that may be useful internally but is not safe to present as final customer-ready output.

### Geometry-only

Mesh exists, but there is no usable UV/embedded texture.

### Texture degraded

Texturing partially failed or output lacks complete embedded texture semantics.

### Recapture required

The capture is too weak or incomplete. User/operator should repeat capture with guidance.

### Training manifest

Privacy-safe structured record of a capture/reconstruction outcome for future ML/fine-tuning use.

### AI fallback

Generated estimate from TRELLIS/Meshy or similar. Useful for preview/draft, not automatically a faithful scan.

---

## 24. Final Operator Rule

The operational success signal is not:

```text
A GLB file exists.
```

The success signal is:

```text
The asset passed customer-ready validation, has usable UV + embedded texture,
has sane geometry, no severe contamination, and produced a training manifest.
```