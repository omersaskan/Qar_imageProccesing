# AI 3D Phases B-G Specification

This document is the source of truth for Phase B through Phase G of the Meshysiz Product Asset Factory AI 3D workstream.

## Core principle

Photogrammetry remains the production source of truth.

AI, SAM3D, Meshy, or other generative 3D outputs are initially only:

- review-only
- diagnostic
- comparison artifacts
- manual-review required
- non-production

AI outputs must never automatically:

- clear `recapture_required`
- bypass coverage gates
- bypass `quality_manifest` gates
- bypass validation gates
- set `publish_state=published`
- change registry active pointer
- overwrite photogrammetry manifest
- become `production_pass`

---

## Phase B — Safe No-Op AI 3D Proxy

### Goal

Phase B does not implement real SAM3D or Meshy generation.

The goal is to install a safe AI proxy infrastructure that can:

- create an AI proxy preview attempt for a session
- write metadata
- mark everything as review-only
- avoid touching the production pipeline

Provider output may be no-op in this phase. The important deliverable is the safety contract.

### Module structure

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
```

### Models

#### `AIProxyRequest`

Required fields:

```python
session_id: str
product_id: str
product_profile: str | None
source_frame_path: str | None
source_mask_path: str | None
quality_manifest_path: str | None
coverage_report_path: str | None
requested_provider: str
```

#### `AIProxyResult`

Required fields:

```python
success: bool
provider_name: str
geometry_source: str
metadata: AIProxyMetadata
preview_mesh_path: str | None
preview_mesh_bytes: bytes | None
failure_reason: str | None
generation_time_sec: float | None
estimated_cost: float | None
```

#### `AIProxyMetadata`

Required fields:

```python
session_id: str
product_id: str | None

ai_proxy_attempted: bool = True
ai_generated: bool = False
preview_artifact_created: bool = False

geometry_source: str = "none"  # none | meshy | sam3d
provider_name: str = "none"
provider_version: str | None = None

production_status: str = "review_ready"
requires_manual_review: bool = True
may_override_recapture_required: bool = False

source_frame_path: str | None = None
source_mask_path: str | None = None

observed_surface_from_capture: float | None = None
ai_inferred_surface: float | None = None
estimated_visible_surface: float | None = None
proxy_confidence: float = 0.0

generation_time_sec: float | None = None
estimated_cost: float | None = None
failure_reason: str | None = None
limitations: list[str] = []
created_at: datetime
```

### Metadata invariants

Every metadata object must enforce:

```python
production_status == "review_ready"
requires_manual_review is True
may_override_recapture_required is False
```

If provider is disabled:

```python
ai_proxy_attempted = True
ai_generated = False
preview_artifact_created = False
failure_reason = "AI preview disabled"  # or equivalent
```

If provider creates a preview mesh:

```python
ai_proxy_attempted = True
ai_generated = True
preview_artifact_created = True
requires_manual_review = True
may_override_recapture_required = False
```

### Provider interface

```python
class AI3DProvider:
    name: str

    def is_available(self) -> bool:
        ...

    def generate_proxy(self, request: AIProxyRequest) -> AIProxyResult:
        ...
```

### Provider behavior

#### `none_provider.py`

- Always safe no-op.
- Does not create a mesh.
- Writes/returns metadata.
- Does not touch session state.

#### `meshy_provider.py`

Phase B must not call the real Meshy API.

Behavior:

- `MESHY_ENABLED=false` returns safe failure metadata.
- Missing `MESHY_API_KEY` returns safe failure metadata.
- No real external API call.

#### `sam3d_provider.py`

Phase B must not run real SAM3D inference.

Behavior:

- `SAM3D_ENABLED=false` returns safe failure metadata.
- Missing checkpoint returns safe failure metadata.
- No real model execution.

### Endpoint

Add:

```text
POST /api/sessions/{session_id}/ai-proxy-preview
```

Endpoint must:

1. validate `session_id` with `validate_identifier(session_id)`
2. load session with `session_manager.get_session(session_id)`
3. create reports dir if needed
4. choose first existing frame from `session.extracted_frames`
5. fallback to `captures/{session_id}/frames/*.jpg` if needed
6. read `ar_quality_manifest.json` if present
7. read `coverage_report.json` if present
8. find mask if available
9. select provider from settings
10. create no-op metadata if provider disabled
11. write `reports/ai_proxy_metadata.json`
12. write `reports/ai_proxy_preview.glb` only if provider returns mesh bytes/path
13. return metadata

Endpoint must not:

- change `session.status`
- change `session.publish_state`
- call `registry.register_asset()`
- call `registry.publish_asset()`
- call `registry.set_active_version()`
- overwrite `coverage_score`
- overwrite `reconstruction_manifest_path`
- overwrite photogrammetry manifest
- clear `recapture_required`

### Phase B tests

Add:

```text
tests/test_ai_proxy_safety.py
tests/test_ai_provider_contract.py
```

Required tests:

```python
def test_ai_provider_disabled_returns_disabled_metadata(): ...
def test_ai_proxy_metadata_requires_manual_review(): ...
def test_ai_proxy_may_override_recapture_required_is_false(): ...
def test_ai_proxy_failure_does_not_fail_session(): ...
def test_recapture_required_session_remains_recapture_required(): ...
def test_ai_proxy_does_not_overwrite_photogrammetry_manifest(): ...
def test_ai_proxy_does_not_register_asset(): ...
def test_ai_proxy_does_not_write_registry_active_pointer(): ...
def test_meshy_without_api_key_fails_safely(): ...
def test_sam3d_disabled_fails_safely(): ...
def test_ai_proxy_endpoint_writes_only_reports_dir(): ...
def test_ai_proxy_endpoint_does_not_change_session_json_status(): ...
```

### Completion criteria

Phase B is complete only when:

- AI proxy endpoint exists
- metadata is written under `reports/`
- no-op provider works safely
- Meshy/SAM3D scaffolds return safe failure
- session status does not change
- `publish_state` does not change
- registry active pointer does not change
- photogrammetry manifest is not overwritten
- all AI safety tests pass

---

## Phase C — AI Provider Abstraction

### Goal

Expand the provider architecture for future Meshy and SAM3D support, without production publishing.

Providers:

- `none`
- `meshy`
- `sam3d`

### Config

Settings and `.env.example` must include:

```env
AI_3D_PROVIDER=none
AI_3D_PREVIEW_ENABLED=false

SAM3D_ENABLED=false
SAM3D_OUTPUT_FORMAT=glb
SAM3D_REQUIRE_REVIEW=true
SAM3D_DEVICE=cuda
SAM3D_CHECKPOINT=
SAM3D_CONFIG=

MESHY_ENABLED=false
MESHY_API_KEY=
MESHY_REQUIRE_REVIEW=true
```

### Provider contract

Every provider must guarantee:

- returns metadata
- does not crash endpoint on failure
- does not log secrets
- does not change session status
- does not write to registry
- returns `production_status=review_ready`
- returns `requires_manual_review=true`
- returns `may_override_recapture_required=false`

### Meshy provider behavior

Even in the first real integration:

- `MESHY_ENABLED=false` -> disabled metadata
- empty `MESHY_API_KEY` -> safe failure
- rate limit -> safe failure
- timeout -> safe failure
- API failure -> safe failure
- success -> review-only preview artifact

Allowed writes:

```text
reports/ai_proxy_metadata.json
reports/ai_proxy_preview.glb
```

Forbidden writes:

```text
registry/blobs/active
registry/meta
production manifest
```

### SAM3D provider behavior

In the first real integration:

- `SAM3D_ENABLED=false` -> disabled metadata
- missing checkpoint -> safe failure
- CUDA missing -> CPU fallback only if explicitly allowed by config
- timeout -> safe failure
- success -> review-only preview artifact

SAM3D must not run per-frame on mobile. It must run backend-side using selected accepted frames.

### Phase C tests

```python
def test_provider_none_contract(): ...
def test_provider_meshy_without_api_key_safe_failure(): ...
def test_provider_sam3d_disabled_safe_failure(): ...
def test_provider_result_always_review_ready(): ...
def test_provider_never_sets_production_pass(): ...
def test_provider_does_not_log_api_key(): ...
def test_provider_timeout_returns_metadata(): ...
```

---

## Phase D — Evaluation Harness

### Goal

Compare photogrammetry, Meshy, SAM3D, and AI proxy outputs.

Evaluation is report-only. It must never affect production status.

### Module structure

```text
modules/evaluation/
  __init__.py
  metrics.py
  compare_outputs.py
  report_writer.py
```

### Inputs

- photogrammetry manifest
- AI proxy metadata
- AI preview mesh path
- validation report
- coverage report
- quality manifest

### Outputs

```text
reports/ai_eval_report.json
reports/ai_eval_summary.md
```

### Metrics

- visual quality
- geometry fidelity
- logo/text preservation
- texture realism
- mobile AR performance
- polycount
- file size
- generation time
- cost
- manual review result
- coverage agreement
- failure reason

### Evaluation must never

- change `session.status`
- change `publish_state`
- change registry pointer
- change validation decision
- clear `recapture_required`

### Phase D tests

```python
def test_eval_report_generated_from_mock_outputs(): ...
def test_missing_ai_preview_does_not_fail_evaluation(): ...
def test_evaluation_cannot_set_production_pass(): ...
def test_evaluation_does_not_modify_session(): ...
def test_evaluation_does_not_touch_registry(): ...
```

---

## Phase E — Manual Review Workflow

### Goal

Allow humans to review AI previews.

This is still not production publishing. It is only a review workflow.

### Review states

- `review_ready`
- `approved_for_internal_preview`
- `rejected`
- `promoted_to_manual_production_candidate`

### Endpoints

```text
POST /api/sessions/{session_id}/ai-review/approve
POST /api/sessions/{session_id}/ai-review/reject
```

### Audit log

Write:

```text
reports/ai_review_audit.jsonl
```

Each line:

```json
{
  "session_id": "...",
  "reviewer": "...",
  "timestamp": "...",
  "action": "approve|reject",
  "reason": "...",
  "provider": "meshy|sam3d|none",
  "asset_path": "...",
  "previous_state": "...",
  "new_state": "..."
}
```

### Rules

- Manual approval must be an explicit API action.
- Approval must write an audit event.
- Rejection must write an audit event.
- Unreviewed AI preview cannot publish.
- `approved_for_internal_preview` is not production.
- `promoted_to_manual_production_candidate` is still not active version.
- Manual review must not automatically connect to production publish.

Review endpoints must never:

- call `registry.publish_asset()`
- call `registry.set_active_version()`
- set `publish_state=published`
- set `session.status=PUBLISHED`

### Phase E tests

```python
def test_ai_review_approval_creates_audit_event(): ...
def test_ai_review_rejection_creates_audit_event(): ...
def test_unreviewed_ai_preview_cannot_publish(): ...
def test_approved_preview_still_not_active_version(): ...
def test_review_endpoint_does_not_clear_recapture_required(): ...
def test_review_endpoint_does_not_overwrite_photogrammetry_manifest(): ...
```

---

## Phase F — UI + WebGL Review Experience

Phase F has its own detailed document:

- [Phase F WebGL Review Experience](PHASE_F_WEBGL_REVIEW.md)

High-level goal:

- expose AI preview in the dashboard
- keep photogrammetry and AI preview visually separate
- add WebGL-based review tools
- add coverage debug tools
- add AI vs photogrammetry comparison tools

UI must show warnings:

- `AI-generated preview. Not production coverage.`
- `Does not bypass recapture.`
- `Manual review required.`
- `Review-only artifact.`

UI must not modify:

- `GateValidator`
- `MediaRecorder`
- `finishRecording`
- quality manifest
- capture HUD accept/reject logic
- production asset card publish state

---

## Phase G — Production Policy Guardrails

### Goal

Add final hard production guards that prevent AI output from leaking into production.

### Guardrail points

- `AssetRegistry.publish_asset()`
- `AssetRegistry.set_active_version()`
- `PackagePublisher.publish_package()` if present
- `worker._handle_publish()`

### Automatic publish is forbidden if

```python
metadata.ai_generated is True
metadata.requires_manual_review is True
metadata.geometry_source != "photogrammetry"
metadata.production_status != "production_pass"
metadata.may_override_recapture_required is False and session.status == RECAPTURE_REQUIRED
```

### Forbidden behavior

- AI preview cannot change `RECAPTURE_REQUIRED` to `VALIDATED`.
- AI preview cannot set `publish_state=published`.
- AI preview cannot update active registry pointer.
- AI preview cannot overwrite photogrammetry manifest.
- AI preview cannot write into `registry/blobs` as active asset.
- AI preview cannot mark itself `production_pass`.

### Phase G tests

```python
def test_ai_generated_asset_cannot_publish_automatically(): ...
def test_requires_manual_review_cannot_publish_automatically(): ...
def test_non_photogrammetry_geometry_cannot_be_active_by_default(): ...
def test_ai_proxy_cannot_clear_recapture_required(): ...
def test_ai_proxy_cannot_set_publish_state_published(): ...
def test_ai_proxy_cannot_update_active_registry_pointer(): ...
def test_ai_proxy_cannot_overwrite_photogrammetry_manifest(): ...
```

---

## Implementation order

### PR 1 — Phase B Safe Shell

- `AIProxyMetadata`
- `AIProxyRequest`
- `AIProxyResult`
- `AI3DProvider` interface
- `none_provider`
- safe Meshy scaffold
- safe SAM3D scaffold
- `proxy_service`
- `POST /api/sessions/{session_id}/ai-proxy-preview`
- `tests/test_ai_proxy_safety.py`
- `tests/test_ai_provider_contract.py`

No real API calls.

### PR 2 — Provider Contract Expansion

- provider availability checks
- timeout handling
- provider version/cost/time metadata
- safe failure behavior
- no secret logging tests

No real Meshy/SAM3D unless heavily mocked or explicitly isolated.

### PR 3 — Evaluation Harness

- `modules/evaluation/`
- `ai_eval_report.json`
- `ai_eval_summary.md`
- mock comparison tests

### PR 4 — Manual Review

- AI review approve/reject endpoints
- `ai_review_audit.jsonl`
- review state transitions
- tests

### PR 5 — UI + WebGL Preview

- `GET /api/config`
- review assets endpoints
- WebGL review viewer
- AI preview card
- warnings
- coverage debug viewer
- AI vs photogrammetry comparison

### PR 6 — Hard Guardrails

- registry/publisher/worker publish safety hardening
- AI generated cannot auto-publish
- requires review cannot auto-publish
- non-photogrammetry cannot become active-by-default
