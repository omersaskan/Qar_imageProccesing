# AI 3D Safety Invariants

This document defines the hard safety contract for all AI, SAM3D, Meshy, and generative 3D work in Meshysiz Product Asset Factory.

## Non-negotiable production principle

Photogrammetry is the production source of truth.

AI-generated or generative 3D artifacts are allowed only as:

- review-only artifacts
- diagnostics
- comparison artifacts
- manual-review candidates
- non-production previews

They are not production outputs by default.

## Global forbidden actions

AI proxy, Meshy, SAM3D, evaluation, review UI, and WebGL comparison flows must never automatically:

- clear `recapture_required`
- override coverage gates
- override `quality_manifest` gates
- override validation gates
- set `session.status=PUBLISHED`
- set `publish_state=published`
- call `AssetRegistry.publish_asset()`
- call `AssetRegistry.set_active_version()`
- write to `registry/active`
- write AI previews into `registry/blobs` as active assets
- overwrite the photogrammetry reconstruction manifest
- overwrite photogrammetry validation reports
- mark AI output as `production_pass`

## Metadata invariants

For every AI proxy metadata object, these fields are invariant:

```json
{
  "production_status": "review_ready",
  "requires_manual_review": true,
  "may_override_recapture_required": false
}
```

These values must hold even when:

- provider is disabled
- provider fails
- provider times out
- Meshy API key is missing
- SAM3D checkpoint is missing
- AI preview mesh is successfully created

## Allowed write locations

AI proxy and review artifacts may write only to session report paths such as:

```text
data/captures/{session_id}/reports/ai_proxy_metadata.json
data/captures/{session_id}/reports/ai_proxy_preview.glb
data/captures/{session_id}/reports/ai_eval_report.json
data/captures/{session_id}/reports/ai_eval_summary.md
data/captures/{session_id}/reports/ai_review_audit.jsonl
data/captures/{session_id}/reports/capture_debug_manifest.json
```

They must not write into:

```text
data/registry/meta
data/registry/active
data/registry/blobs
reconstruction job manifests
production validation reports
```

## Session state rules

AI proxy, evaluation, review, and WebGL UI endpoints must be read-only or report-only with respect to session production state.

They must not change:

- `session.status`
- `session.publish_state`
- `session.coverage_score`
- `session.reconstruction_manifest_path`
- `session.validation_report_path`
- `session.export_blob_path`
- `session.asset_id`
- `session.asset_version`

Manual review endpoints may update AI review metadata or audit logs, but they must still not publish anything.

## Provider rules

All providers must:

- return metadata on failure
- avoid unhandled exceptions escaping to the endpoint
- avoid logging API keys or secrets
- avoid writing directly to registry paths
- avoid changing session state
- return `production_status=review_ready`
- return `requires_manual_review=true`
- return `may_override_recapture_required=false`

## UI rules

UI and WebGL viewers must remain non-blocking and review-only.

They must not modify:

- `GateValidator`
- `MediaRecorder`
- `finishRecording`
- quality manifest creation
- capture HUD accept/reject logic
- production asset publish state

AI preview UI must clearly show:

- `AI-generated preview. Not production coverage.`
- `Does not bypass recapture.`
- `Manual review required.`
- `Review-only artifact.`

## Completion rule

No phase is complete unless the new safety tests pass and the existing upload/capture/reconstruction regression tests still pass.
