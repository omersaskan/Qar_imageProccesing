# Reconstruction Reliability Plan

## 1. Core Engine Decision

The primary reconstruction engine remains:

```text
COLMAP + OpenMVS
```

COLMAP is used for feature extraction, matching, sparse reconstruction, and dense reconstruction. OpenMVS is used for mesh reconstruction and texturing where available.

The long-term system should not depend exclusively on external paid asset APIs. External AI systems may be used as draft or fallback engines, but not as the primary source of truth for faithful scans.

---

## 2. Required Engine Configuration

### Required Environment Variables

```bash
RECON_ENGINE_PATH=C:\colmap\colmap.exe
OPENMVS_BIN_PATH=C:\openmvs\bin
RECON_PIPELINE=colmap_openmvs
RECON_USE_GPU=true
REQUIRE_TEXTURED_OUTPUT=true
RECON_FALLBACK_STEPS=default,denser_frames
RECON_UNMASKED_FALLBACK_ENABLED=false
```

### OpenMVS Environment Compatibility

The canonical runtime value should be:

```text
settings.openmvs_path
```

Backward-compatible environment variables:

```text
OPENMVS_BIN_PATH
OPENMVS_BIN
```

If both exist, `OPENMVS_BIN_PATH` should be treated as canonical.

---

## 3. Pilot / Production Guards

The following must be enforced in `pilot` and `production`:

- `RECON_PIPELINE=simulated` is forbidden.
- Missing COLMAP binary should make readiness fail.
- Missing OpenMVS binary should make readiness fail or warn depending on required texture policy.
- Missing ML dependencies should block ingestion.
- Texture-less output must not be customer-ready.
- Geometry-only output must not be customer-ready.

---

## 4. Attempt Scoring

The reconstruction attempt scoring system must consider both geometry and deliverable asset quality.

### Geometry Metrics

- registered images
- sparse points
- dense points fused
- mesher used
- mesh vertex count
- mesh face count

### Texture / Material Metrics

- has UV
- has material
- has embedded texture
- texture integrity status
- material semantic status
- texturing status

### Cleanup / Contamination Metrics

- component count
- bbox sanity
- table/support contamination
- product geometry removed
- selected component confidence

### Suggested Scoring Direction

Positive:

- many registered images
- stable sparse model
- dense point cloud
- valid mesh
- real OpenMVS texture
- UV present
- embedded texture present
- sane bbox
- low contamination

Negative:

- weak sparse model
- failed mesher
- geometry-only output
- missing UV
- missing embedded texture
- too many components
- suspected table/support contamination
- cleanup removed too much geometry

---

## 5. Texture and UV Rules

A mesh can be geometrically valid but still not be customer-ready.

Rules:

1. Texture path alone is not enough.
2. Mesh must have UV coordinates.
3. GLB must have embedded texture.
4. Cleanup must preserve UV/material or trigger re-texturing.
5. Exported GLB must be reloaded and inspected.
6. Customer-ready validation must block geometry-only assets.

---

## 6. Failure States

The system should distinguish these states:

| State | Meaning |
|---|---|
| `recapture_required` | Input capture is not sufficient |
| `reconstruction_failed` | Engine failed or output too weak |
| `texture_degraded` | Geometry exists but texturing is incomplete |
| `geometry_only` | Mesh exists without usable UV/texture |
| `uv_only` | UV exists but embedded texture missing |
| `draft_asset` | Internal asset, not final customer-ready |
| `customer_ready` | Passed all required quality gates |

---

## 7. Fallback Strategy

### Primary

```text
Guided capture -> COLMAP + OpenMVS -> textured GLB
```

### Fallbacks

| Fallback | Purpose |
|---|---|
| Denser frame extraction | More input frames for weak capture |
| Controlled unmasked fallback | When masks hurt reconstruction |
| COLMAP dense fallback | Internal geometry fallback when texture is not required |
| AI draft fallback | Quick preview or non-critical generated estimate |
| Recapture | When capture is fundamentally weak |

---

## 8. Acceptance Criteria

The reliability migration is successful when:

- COLMAP and OpenMVS readiness are checked.
- Simulated reconstruction is blocked in pilot/production.
- Reconstruction attempt scoring considers texture/UV.
- GLB export re-inspects embedded texture.
- Geometry-only assets cannot be customer-ready.
- Validation clearly reports material semantic status.
- Texture-degraded outputs are stored as draft/degraded, not final.