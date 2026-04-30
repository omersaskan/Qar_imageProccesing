# AI 3D Agent Implementation Plan

This document is written for implementation agents. Follow this order strictly.

Do not implement later-phase behavior in earlier PRs. Keep every PR small, safe, and testable.

## Global instruction

Read these docs before changing code:

1. `docs/AI_3D_SAFETY_INVARIANTS.md`
2. `docs/AI_3D_PHASES_B_TO_G.md`
3. `docs/PHASE_F_WEBGL_REVIEW.md`

Photogrammetry remains the production source of truth.

AI, SAM3D, Meshy, and other generative 3D outputs are review-only unless a future explicit production policy says otherwise.

## Forbidden unless explicitly requested in the current PR

Do not:

- change `GateValidator`
- change `MediaRecorder`
- change `finishRecording`
- change upload normalization
- change `quality_manifest` validation
- change production publish rules
- implement real Meshy API calls in Phase B
- implement real SAM3D inference in Phase B
- write AI outputs into registry active paths
- set `publish_state=published` from AI flows
- clear `recapture_required` from AI flows

## PR 1 — Phase B Safe Shell

### Goal

Install safe AI proxy infrastructure with no real external AI generation.

### Add files

```text
modules/ai_3d_proxy/
  __init__.py
  metadata.py
  provider_base.py
  proxy_service.py
  providers/
    __init__.py
    none_provider.py
    meshy_provider.py
    sam3d_provider.py

tests/test_ai_proxy_safety.py
tests/test_ai_provider_contract.py
```

### Add endpoint

```text
POST /api/sessions/{session_id}/ai-proxy-preview
```

### Required behavior

- validate `session_id`
- load session
- resolve one existing frame
- read `ar_quality_manifest.json` if present
- read `coverage_report.json` if present
- select provider from settings
- write `reports/ai_proxy_metadata.json`
- write `reports/ai_proxy_preview.glb` only if provider returns preview mesh
- return metadata

### Must not do

- change `session.status`
- change `session.publish_state`
- call `registry.register_asset()`
- call `registry.publish_asset()`
- call `registry.set_active_version()`
- overwrite photogrammetry manifest
- clear `recapture_required`
- overwrite `coverage_score`

### Verification

Run:

```bash
node --check ui/app.js
python -m pytest tests/test_hardening_v2.py tests/test_fix_regressions.py tests/test_hardened_upload.py tests/test_ar_mask_preview.py
python -m pytest tests/test_ai_proxy_safety.py
python -m pytest tests/test_ai_provider_contract.py
```

Do not claim Phase B complete unless all tests pass.

## PR 2 — Phase C Provider Contract Expansion

### Goal

Harden provider abstraction and metadata reporting.

### Add or improve

- provider availability checks
- timeout handling
- provider version metadata
- generation time metadata
- estimated cost metadata
- safe failure behavior
- no-secret logging tests

### Still avoid

- production publish
- registry writes
- automatic active version changes
- real provider calls unless isolated and explicitly requested

### Tests

Add tests for:

- none provider contract
- Meshy missing API key
- SAM3D disabled
- timeout safe failure
- result always review ready
- provider never sets production pass
- API key not logged

## PR 3 — Phase D Evaluation Harness

### Goal

Add report-only comparison between photogrammetry and AI preview outputs.

### Add files

```text
modules/evaluation/
  __init__.py
  metrics.py
  compare_outputs.py
  report_writer.py
```

### Outputs

```text
reports/ai_eval_report.json
reports/ai_eval_summary.md
```

### Rules

Evaluation must not change:

- session status
- publish state
- registry pointer
- validation decision
- recapture state

### Tests

Add tests for:

- report generation from mock outputs
- missing AI preview does not fail evaluation
- evaluation cannot set production pass
- evaluation does not modify session
- evaluation does not touch registry

## PR 4 — Phase E Manual Review

### Goal

Add explicit human review workflow for AI previews.

### Endpoints

```text
POST /api/sessions/{session_id}/ai-review/approve
POST /api/sessions/{session_id}/ai-review/reject
```

### Output

```text
reports/ai_review_audit.jsonl
```

### Rules

Manual review may update AI review state and audit logs only.

It must not:

- publish asset
- set active version
- set `publish_state=published`
- set `session.status=PUBLISHED`
- clear recapture required

### Tests

Add tests for:

- approval audit event
- rejection audit event
- unreviewed preview cannot publish
- approved preview still not active version
- review endpoint does not clear recapture required
- review endpoint does not overwrite photogrammetry manifest

## PR 5A — Phase F.0 Backend Review Endpoints

### Goal

Add read-only endpoints needed by UI/WebGL review.

### Endpoints

```text
GET /api/config
GET /api/sessions/{session_id}/review-assets
GET /api/sessions/{session_id}/coverage-debug
GET /api/sessions/{session_id}/comparison-data
```

### Rules

Endpoints are read-only. They must not change session or registry state.

### Tests

Add endpoint tests for:

- config returns feature flags
- review assets returns distinct photogrammetry and AI blocks
- coverage debug handles missing manifest gracefully
- comparison data handles missing AI preview gracefully

## PR 5B — Phase F.1 WebGL Review Viewer

### Goal

Add browser GLB viewer with distinct photogrammetry and AI preview cards.

### Recommended files

```text
ui/webgl/
  viewer_core.js
  review_viewer.js
  overlays.js
  loaders.js
  ui_panels.js
```

### Required features

- GLB viewer
- photogrammetry card
- AI preview card
- bbox overlay
- pivot overlay
- ground plane overlay
- texture toggle
- wireframe toggle

### Required warnings

AI preview must show:

```text
AI-generated preview. Not production coverage.
Does not bypass recapture.
Manual review required.
Review-only artifact.
```

### Tests

Add DOM/helper/Playwright tests if available.

## PR 5C — Phase F.2 Capture Coverage Debug Viewer

### Goal

Add post-session coverage visualization.

### Required features

- accepted frame camera ring
- missing angles
- coverage heatmap
- frame confidence markers
- rejection reason visualization

### Optional artifact

```text
reports/capture_debug_manifest.json
```

This is debug-only and must not be a production gate.

## PR 5D — Phase F.3 AI vs Photogrammetry Comparison

### Goal

Add side-by-side comparison viewer.

### Required features

- synchronized side-by-side viewers
- silhouette comparison
- warning labels
- read-only manual review notes
- non-blocking behavior when AI preview is missing

## PR 6 — Phase G Hard Guardrails

### Goal

Prevent AI output from becoming production accidentally.

### Harden these points

- `AssetRegistry.publish_asset()`
- `AssetRegistry.set_active_version()`
- `PackagePublisher.publish_package()` if present
- `worker._handle_publish()`

### Forbidden automatic publish when

```python
metadata.ai_generated is True
metadata.requires_manual_review is True
metadata.geometry_source != "photogrammetry"
metadata.production_status != "production_pass"
metadata.may_override_recapture_required is False and session.status == RECAPTURE_REQUIRED
```

### Tests

Add tests for:

- AI generated asset cannot publish automatically
- requires manual review cannot publish automatically
- non-photogrammetry geometry cannot become active by default
- AI proxy cannot clear recapture required
- AI proxy cannot set publish state published
- AI proxy cannot update active registry pointer
- AI proxy cannot overwrite photogrammetry manifest

## Final verification before merging any phase

At minimum run:

```bash
node --check ui/app.js
python -m pytest tests/test_hardening_v2.py tests/test_fix_regressions.py tests/test_hardened_upload.py tests/test_ar_mask_preview.py
```

Also run all new tests introduced by the PR.

If a test file does not exist yet, do not silently ignore it in completion notes. State that it is not present and create it when relevant to the PR.
