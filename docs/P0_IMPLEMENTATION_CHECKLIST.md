# P0 Implementation Checklist

## 1. Environment / Settings

- [ ] Harmonize `OPENMVS_BIN_PATH` and `OPENMVS_BIN`.
- [ ] Use `settings.openmvs_path` as the canonical OpenMVS setting.
- [ ] Add OpenMVS binary readiness probe.
- [ ] Check `InterfaceCOLMAP`.
- [ ] Check `TextureMesh`.
- [ ] Check `DensifyPointCloud` where applicable.
- [ ] Check `ReconstructMesh` where applicable.
- [ ] Clarify pilot/prod default pipeline.
- [ ] Prefer `RECON_PIPELINE=colmap_openmvs` in pilot/prod.
- [ ] Enforce no simulated reconstruction in pilot/prod.
- [ ] Add readiness issue if simulated pipeline is configured in pilot/prod.

---

## 2. GLB Export

- [ ] Strengthen UV/texture guard.
- [ ] If texture exists but UV is missing, report explicit warning.
- [ ] Do not mark texture application successful unless exported GLB inspection confirms embedded texture.
- [ ] Force PBR `TextureVisuals` fallback when UV + texture exist but existing material does not support texture slots.
- [ ] Export and re-inspect GLB.
- [ ] Persist texture integrity status.
- [ ] Persist material semantic status.
- [ ] Preserve existing PBR maps when possible.

---

## 3. Reconstruction Scoring

- [ ] Add UV score.
- [ ] Add material score.
- [ ] Add embedded texture score.
- [ ] Add texture integrity score.
- [ ] Add material semantic score.
- [ ] Add component count penalty.
- [ ] Add bbox sanity score.
- [ ] Add table/support contamination penalty.
- [ ] Penalize geometry-only outputs.
- [ ] Penalize missing embedded texture.
- [ ] Prefer textured OpenMVS output when customer-ready mode is enabled.

---

## 4. API / UI

- [ ] Add upload preflight metadata.
- [ ] Validate file size.
- [ ] Validate video duration.
- [ ] Validate FPS.
- [ ] Validate resolution.
- [ ] Validate basic readability.
- [ ] Add codec/orientation best-effort checks.
- [ ] Make `API_BASE` configurable in UI.
- [ ] Keep upload as admin/internal path.
- [ ] Document that customer-facing capture should use guided capture.

---

## 5. Validation / Publish

- [ ] Geometry-only GLB must not be customer-ready.
- [ ] UV-only GLB must not be customer-ready.
- [ ] Missing embedded texture must block customer-ready publish.
- [ ] Draft/degraded state must be separate from customer-ready state.
- [ ] Validation report should include material semantic status.
- [ ] Validation report should include texture integrity status.
- [ ] Publish logic should respect customer-ready requirements.

---

## 6. Training Data

- [ ] Add training manifest schema.
- [ ] Add label taxonomy.
- [ ] Add training manifest builder.
- [ ] Add training registry index.
- [ ] Add consent status field.
- [ ] Add `eligible_for_training` field.
- [ ] Hash product identifiers in training manifest.
- [ ] Generate manifest for completed sessions.
- [ ] Generate manifest for failed sessions.
- [ ] Generate manifest for recapture sessions.
- [ ] Manifest generation must be best-effort and must not break publish.
- [ ] Add internal API endpoints for training manifests.

---

## 7. Tests

- [ ] Settings test for OpenMVS env compatibility.
- [ ] Settings test for simulated pipeline blocked in pilot/prod.
- [ ] Readiness test for missing OpenMVS binaries.
- [ ] GLB exporter test for UV missing + texture present.
- [ ] GLB exporter test for fallback PBR material.
- [ ] Validation test for geometry-only asset.
- [ ] Validation test for textured customer-ready asset.
- [ ] Upload preflight test for invalid video.
- [ ] Training manifest test with missing reports.
- [ ] Training manifest test for customer-ready asset.
- [ ] Training manifest test for geometry-only asset.
- [ ] Registry append/update test.