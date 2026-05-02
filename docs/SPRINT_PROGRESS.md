# Sprint Progress (5–8)

> Implementation-only mode. Tests in Sprint 9.

---

## Sprint 5 — COLMAP Pose-Backed Coverage Matrix

**Yeni dosyalar:**
- `modules/reconstruction_engine/colmap_sparse_parser.py` — cameras.txt / images.txt text parser
- `modules/reconstruction_engine/pose_geometry.py` — qvec→R, camera_center, cartesian→spherical
- `modules/reconstruction_engine/pose_coverage_matrix.py` — 3×8 grid (3 elevation × 8 azimuth sectors)
- `modules/reconstruction_engine/orbit_validation.py` — coverage_ratio / azimuth_span / elevation_spread verdict
- `modules/reconstruction_engine/pose_feedback.py` — top-level orchestrator → manifest block

**Manifest block:** `pose_backed_coverage` (status, coverage, orbit_validation, sparse_model_dir)

**Flag:** `POSE_BACKED_COVERAGE_ENABLED=false` (default)

**Bilinçli sınırlar:** intrinsics feed kapalı; Sprint 4.6 fallback loop coverage path'i henüz bağlanmadı.

---

## Sprint 6 — Blender Headless Cleanup / Export Worker

**Yeni dosyalar:**
- `modules/asset_cleanup/__init__.py`
- `modules/asset_cleanup/mesh_normalization.py` — NormalizationConfig dataclass
- `modules/asset_cleanup/blender_script_generator.py` — Blender Python script template
- `modules/asset_cleanup/blender_headless_worker.py` — subprocess runner, graceful unavailable
- `modules/export_pipeline/glb_export_manifest.py` — BlenderWorkerResult → manifest block

**Manifest block:** `blender_cleanup` (status, output_glb, blender_version, elapsed_seconds, reason)

**Flags:** `BLENDER_CLEANUP_ENABLED=false`, `BLENDER_CLEANUP_DECIMATE_ENABLED=false`, `BLENDER_CLEANUP_DECIMATE_RATIO=0.5`

**Bilinçli sınırlar:** Blender binary yoksa `status=unavailable`; hiçbir exception yayılmaz. Decimation default kapalı.

---

## Sprint 7 — glTF-Transform + Khronos Validator Publish Gate

**Yeni dosyalar:**
- `modules/export_pipeline/gltf_transform_optimizer.py` — `gltf-transform optimize` subprocess wrapper
- `modules/qa_validation/gltf_validator.py` — Khronos `gltf_validator` JSON output parser
- `modules/qa_validation/ar_asset_gate.py` — optimizer + validator → pass/review/reject verdict

**Manifest blocks:** `gltf_optimization`, `gltf_validation`, `ar_asset_gate`

**Flags:** `GLTF_OPTIMIZATION_ENABLED=false`, `GLTF_VALIDATION_ENABLED=false`, `GLTF_VALIDATION_REJECT_ON_ERROR=true`

**Bilinçli sınırlar:** CLI yoksa `status=unavailable`; optimize subcommand path; Draco opt-in kapalı.

---

## Sprint 8 — License Manifest + Provenance

**Yeni dosyalar:**
- `modules/asset_registry/license_manifest.py` — ToolEntry / SourceEntry / LicenseManifest; KNOWN_TOOLS (COLMAP/BSD, OpenMVS/AGPL, Blender/GPL, glTF-Transform/MIT, Khronos/Apache, FFmpeg/LGPL)
- `modules/asset_registry/asset_provenance.py` — ProvenanceStep / AssetProvenance; provenance_from_manifest()

**Output files (when enabled):** `license_manifest.json`, `asset_provenance.json`

**Flags:** `LICENSE_MANIFEST_ENABLED=false`, `PROVENANCE_ENABLED=false`

**OpenMVS AGPL risk note:** `build_license_manifest` auto-appends AGPL warning note when `openmvs` in active_tools.

---

## Genel Durum

| Sprint | Status | Default |
|--------|--------|---------|
| 4.6 | ✅ Complete + 40 tests | off |
| 5 | ✅ Impl only | off (`POSE_BACKED_COVERAGE_ENABLED`) |
| 6 | ✅ Impl only | off (`BLENDER_CLEANUP_ENABLED`) |
| 7 | ✅ Impl only | off (`GLTF_OPTIMIZATION_ENABLED`, `GLTF_VALIDATION_ENABLED`) |
| 8 | ✅ Impl only | off (`LICENSE_MANIFEST_ENABLED`, `PROVENANCE_ENABLED`) |
| 9 | ✅ Complete (QA/Test) | — |

**Kümülatif test sonucu: 286/286 passing, 0 regresyon.**

### Sprint 9 — Test breakdown (103 yeni test)
| Dosya | Testler | Kapsam |
|-------|---------|--------|
| `test_sprint5_pose_pipeline.py` | 35 | COLMAP parser, pose geometry, 3×8 coverage matrix, orbit validation, pose_feedback |
| `test_sprint6_blender_pipeline.py` | 21 | NormalizationConfig, script generator syntax, headless worker (mock), GLB manifest |
| `test_sprint7_gltf_pipeline.py` | 22 | glTF-Transform optimizer (mock), Khronos validator (mock), AR gate logic |
| `test_sprint8_license_provenance.py` | 25 | ToolEntry/AGPL risk, LicenseManifest write, provenance reconstruction from manifest |
