# AI Production Guardrails

These guardrails apply to all Phase B through Phase G work.

## Core principle

Photogrammetry is the only production source of truth.

AI, SAM3D, Meshy, or any generative 3D output must remain review-only until an explicit production policy allows otherwise.

## AI output may be

- review-only
- diagnostic
- comparison artifact
- manually reviewed
- non-production

## AI output must never automatically

- clear `recapture_required`
- bypass coverage gates
- bypass `quality_manifest` gates
- bypass validation gates
- set `publish_state=published`
- update the registry active pointer
- overwrite the photogrammetry manifest
- mark itself as `production_pass`
- write into active production registry/blob locations

## Required metadata invariants

Every AI proxy/review/evaluation artifact must preserve these invariants:

```json
{
  "production_status": "review_ready",
  "requires_manual_review": true,
  "may_override_recapture_required": false
}
```

If a provider is disabled or unavailable, metadata must still be returned. Failure must be safe and non-blocking.

## Forbidden mutations from AI proxy/review/evaluation/UI endpoints

The following must not be mutated by Phase B, C, D, E, or F endpoints:

- `session.status`
- `session.publish_state`
- `session.coverage_score`
- `session.reconstruction_manifest_path`
- registry active pointer files
- registry production metadata
- photogrammetry output manifests
- photogrammetry validation decisions

## Forbidden function calls from AI preview endpoints

AI preview, evaluation, and WebGL review endpoints must not call:

- `AssetRegistry.register_asset()`
- `AssetRegistry.publish_asset()`
- `AssetRegistry.set_active_version()`
- any package publisher that marks output active or published

## Phase G hardening locations

When hard guardrails are implemented, enforce them at:

- `AssetRegistry.publish_asset()`
- `AssetRegistry.set_active_version()`
- `PackagePublisher.publish_package()` if present
- worker publish path such as `worker._handle_publish()`

## Default production policy

By default:

- `metadata.ai_generated is True` cannot publish automatically.
- `metadata.requires_manual_review is True` cannot publish automatically.
- `metadata.geometry_source != "photogrammetry"` cannot become active by default.
- `metadata.production_status != "production_pass"` cannot publish.
- `metadata.may_override_recapture_required is False` cannot clear `RECAPTURE_REQUIRED`.

## Testing expectations

Every phase must include tests proving that AI artifacts cannot:

- move `RECAPTURE_REQUIRED` to `VALIDATED` or `PUBLISHED`
- set `publish_state=published`
- update the active registry pointer
- overwrite photogrammetry manifests
- write active production blobs
- mark themselves as `production_pass`
