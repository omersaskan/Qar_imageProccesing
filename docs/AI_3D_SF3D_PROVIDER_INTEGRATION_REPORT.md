# AI 3D — SF3D Provider Integration Report

## Summary

This report documents the SF3D scaffold implementation. The goal was a production-ready
integration scaffold for Stable Fast 3D that is safe to ship without SF3D installed,
without GPU inference, and without touching the main application Python environment.

## Acceptance Criteria Status

| Criterion | Status |
|---|---|
| `modules/ai_3d_generation/` package exists with all modules | PASS |
| `scripts/sf3d_worker.py` subprocess worker exists | PASS |
| API endpoints `/api/ai-3d/*` implemented | PASS |
| `ui/ai_3d_studio.html` UI page exists | PASS |
| Nav link in `ui/index.html` | PASS |
| All settings flags default to `False` / disabled | PASS |
| SF3D unavailable path is graceful (no crash, structured response) | PASS |
| No real SF3D installation required | PASS |
| Main env is untouched (no sf3d/torch imports in main process) | PASS |
| Tests pass | PASS — 50 new tests, 722 total |
| Report exists | PASS (this document) |

## Files Created / Modified

### New files

| File | Purpose |
|---|---|
| `modules/ai_3d_generation/__init__.py` | Package marker |
| `modules/ai_3d_generation/provider_base.py` | Abstract base + `safe_generate()` |
| `modules/ai_3d_generation/sf3d_provider.py` | SF3D subprocess provider |
| `modules/ai_3d_generation/input_preprocessor.py` | Image → 512×512 square PNG |
| `modules/ai_3d_generation/manifest.py` | Manifest builder + writer |
| `modules/ai_3d_generation/postprocess.py` | GLB postprocess stubs |
| `modules/ai_3d_generation/quality_gate.py` | Verdict assignment |
| `modules/ai_3d_generation/router.py` | input/intent → pipeline decision |
| `modules/ai_3d_generation/pipeline.py` | Full pipeline orchestrator |
| `scripts/sf3d_worker.py` | Isolated worker (runs in SF3D venv) |
| `ui/ai_3d_studio.html` | AI 3D Studio frontend |
| `tests/test_ai_3d_generation.py` | 50 tests covering all modules |
| `docs/SF3D_INTEGRATION_PLAN.md` | Installation + config guide |
| `docs/AI_GENERATED_3D_ARCHITECTURE.md` | Architecture reference |
| `docs/AI_3D_SF3D_PROVIDER_INTEGRATION_REPORT.md` | This document |

### Modified files

| File | Change |
|---|---|
| `modules/operations/settings.py` | Added 14 SF3D + AI 3D flags (all defaulting to False/disabled) |
| `modules/operations/api.py` | Added 5 `/api/ai-3d/*` endpoints + `pydantic.BaseModel` import |
| `ui/index.html` | Added "AI 3D Studio" nav link |

## Test Coverage

```
TestProviderBase            5 tests  — safe_generate: unavailable, exception, normalise, ok, failed
TestSF3DProviderAvailability 5 tests  — disabled, python missing (path/empty), worker missing, all ok
TestSF3DProviderGenerate    8 tests  — timeout, no stdout, bad json, unavailable, failed, ok+exists,
                                       ok+missing, stderr tail
TestQualityGate             6 tests  — unavailable, failed, output missing, review, ok, provenance
TestRouter                  7 tests  — debug→depth_studio, image→ai_3d, video variants, keys, notes
TestInputPreprocessor       6 tests  — center crop, bbox, mask, empty mask, square output, bad image
TestManifest                5 tests  — provenance fields, session_id, status, write+read, nested dir
TestAI3DPipelineIntegration 5 tests  — unavailable, failed, ok→review, provenance, disk write
TestSF3DWorkerDryRun        3 tests  — missing image, existing image, unavailable path (real process)
─────────────────────────────────────────────────────────────────────────────
Total                      50 tests  all PASSED
```

Full suite: **722 passed** (was 672 before this sprint).

## Design Decisions

### Subprocess isolation

SF3D requires PyTorch and model weights. Importing these into the main FastAPI
process would bloat memory, slow startup, and couple the main env to GPU tooling.
The subprocess pattern (identical to Depth Pro) lets the main process stay lean
and keep running even when SF3D is misconfigured.

### Graceful unavailable path

`SF3DProvider.is_available()` checks three conditions before attempting any subprocess:
1. `sf3d_enabled` setting flag
2. `sf3d_python_path` file existence
3. `sf3d_worker_script` file existence

If any fails, `safe_generate()` returns `{"status": "unavailable", ...}` without
spawning a process. The worker itself also catches `ImportError` on `import sf3d`
and exits 0 with `status=unavailable` — so even if the path exists but the package
isn't installed, the system degrades gracefully.

### Provenance invariants

`is_true_scan: false`, `geometry_confidence: "estimated"`, `mode: "ai_generated_3d"`,
and `asset_type: "ai_generated"` are set unconditionally in `build_manifest()` and
cannot be overridden by any provider result. These prevent AI-generated meshes from
being mistaken for photogrammetry scans in downstream consumers.

### Review-by-default policy

Both `SF3D_REQUIRE_REVIEW` and `AI_3D_REQUIRE_REVIEW` default to `True`. This means
all AI-generated outputs land in `status="review"` until an operator explicitly disables
the review requirement — preventing accidental production use of estimated geometry.

## What Is Not Yet Implemented

The following are deferred to later phases and are explicitly out of scope for this scaffold:

- Real SF3D model weights / inference (requires GPU + SF3D venv setup)
- GLB postprocessing (normalisation, optimisation, validation) — stubs exist
- Preview image generation from the GLB
- Video frame extraction direct-from-SF3D (SF3D is image-only; frame selection uses existing depth-studio video router)
- Batch processing endpoint
- Asset registry integration for AI-generated assets
