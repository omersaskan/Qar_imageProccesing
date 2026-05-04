# SF3D Phase 1 Multi-Candidate E2E Report

## Overview

This report documents the implementation, stabilization, and verification of the Phase 1
multi-candidate 3D generation orchestration pipeline using the SF3D (Stable Fast 3D) local provider.

**Scope of Phase 1**: Single-image multi-candidate selection via SF3D.
The pipeline generates multiple 3D candidates from a single image (or from multiple uploaded images
/ video frames), then selects the single best output using heuristics.
**This is single-image multi-candidate selection, not true multi-view 3D reconstruction.**

---

## Commit at Closure

```
HEAD: 0567ccc — test: add comprehensive test suite for AI 3D generation pipeline and provider base classes
```

Additional stabilization commit applied in this session:
```
fix: stabilize AI 3D provider safety and close SF3D phase 1 truthfully
```

---

## Repository State: Mixed Branches

> [!IMPORTANT]
> As of Phase 1 closure, the repository contains **two coexisting bodies of work**:
>
> 1. **SF3D Phase 1 multi-candidate closure** — fully implemented, tested, and stable.
> 2. **External provider / Rodin scaffold** — present but **disabled by default** and **not part of SF3D Phase 1**.
>
> External provider scaffold exists but remains disabled by default and is not part of SF3D Phase 1 closure.

---

## Test Results

### Targeted AI 3D Suite

- **Command**: `py -m pytest tests/test_ai_3d_generation.py -q`
- **Result**: **122 passed** ✅

### External Provider / Security Suite

- **Command**: `py -m pytest tests/test_external_consent_gate.py tests/test_remote_provider_mock.py tests/test_ai_provider_security.py -q`
- **Result**: **14 passed** ✅

### Full Suite

> [!WARNING]
> Full pytest was not completed at original Phase 1 closure time. The statement is therefore:
> **"Full suite not completed; targeted AI 3D suite passed (122 tests)."**
>
> Some unrelated legacy / workflow tests may fail due to missing binaries (COLMAP, OpenMVS, ffmpeg)
> in the local dev environment. These failures are environment-specific and unrelated to the
> AI 3D generation logic.

---

## Security / Stability Guarantees (Stabilization Patch)

The following safety invariants are enforced as of the stabilization patch:

| Invariant | Status |
|---|---|
| `AI_3D_REMOTE_PROVIDERS_ENABLED` defaults to `false` | ✅ |
| `AI_3D_REQUIRE_EXTERNAL_CONSENT` defaults to `true` | ✅ |
| Rodin `is_available()` blocks on global switch before rodin-specific switch | ✅ |
| Unknown provider raises `ValueError("unknown_ai3d_provider:<name>")` | ✅ |
| API maps unknown provider to HTTP 400 `{"error": "unknown_ai3d_provider"}` | ✅ |
| No silent SF3D fallback for unknown providers | ✅ |
| `sanitize_json_like()` available for recursive manifest sanitization | ✅ |
| Error messages / manifests do not expose API keys / Bearer tokens | ✅ |
| Rodin mock mode prohibited outside `local_dev` | ✅ |

---

## Manual E2E Validation Scenarios

### A) Single Image Flow
- **Workflow**: Upload a single image file via UI.
- **Expected Outcome**:
  - `input_mode` = `single_image`
  - `candidate_count` = 1 (or 0 for strict legacy)
  - Output GLB successfully served.
  - Model Viewer successfully loads the model.

### B) Video Top-K Flow
- **Workflow**: Upload a single `.mp4` or video file via UI.
- **Expected Outcome**:
  - `input_mode` = `video`
  - The pipeline extracts top-k sharpest frames.
  - Generates a candidate for each valid frame sequentially.
  - `candidate_count` > 1
  - `selected_candidate_id` points to the winning frame (e.g., `cand_002`).
  - `selected_frame_path` correctly resolves to `derived/selected_frame.jpg`.

### C) Multi-Image Flow
- **Workflow**: Upload 2 or 3 distinct image files simultaneously via UI drag-and-drop.
- **Expected Outcome**:
  - `input_mode` = `multi_image`
  - `candidate_count` equals the number of uploaded images (or capped by `AI_3D_MAX_CANDIDATES`).
  - `selected_candidate_id` points to the highest-scoring candidate.
  - The winner is successfully promoted and served on the `/output` endpoint.

---

## External Provider Scaffold (Rodin)

- Rodin scaffold is present in `modules/ai_3d_generation/rodin_provider.py`
- **All remote providers are disabled by default** (`AI_3D_REMOTE_PROVIDERS_ENABLED=false`)
- Rodin requires: `AI_3D_REMOTE_PROVIDERS_ENABLED=true`, `RODIN_ENABLED=true`, valid `RODIN_API_KEY`
- Mock mode is restricted to `local_dev` environment only
- External providers require explicit `external_provider_consent=true` in every request

---

## Known Limitations

1. **No Background Removal**: Image subjects must be relatively isolated; automatic rembg is deferred to Phase 2.
2. **Quality Ceilings**: Input resolution remains capped by the default pipeline variables (512px). Escalation algorithms are scheduled for later phases.
3. **Sequential Execution Limitation**: SF3D jobs are executed serially to prevent GPU Out-of-Memory exceptions, resulting in longer processing times when submitting multiple inputs.

---

## Phase 2 Gate

> [!IMPORTANT]
> **Phase 2 (rembg/background removal, additional providers) CANNOT begin** until:
> 1. This stabilization commit is merged.
> 2. The targeted AI 3D test suite (122 tests) continues to pass.
> 3. The security/consent suite (14+ tests) continues to pass.
>
> Phase 2 status: **NOT STARTED. Gate open pending commit merge.**
