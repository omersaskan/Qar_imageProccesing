# AI 3D Phase 3D — Defaults, UI, and API E2E Verification Report

**Date:** 2026-05-05
**Commit SHA:** `729aebd` (chore: apply benchmark-driven AI 3D defaults)
**Server:** `127.0.0.1:8002` · SF3D · `wsl_subprocess` · `Ubuntu-24.04`

---

## Summary

| Check | Result |
|---|---|
| Targeted tests | 189 passed, 0 failed |
| Preflight | ok — SF3D available, mode=wsl_subprocess |
| Default E2E (options omitted) | PASS — quality_mode=high, bg_removed=true |
| Override E2E (balanced + bg=false) | PASS — quality_mode=balanced, bg_removed=false |
| UI static verification | PASS — all 5 checks confirmed |
| External providers | Untouched, disabled |

---

## 1. Targeted Test Result

```
189 passed in 11.32s
```

Files tested:
- `tests/test_quality_profiles.py` — includes 4 new Phase 3C default assertions
- `tests/test_phase3a_benchmark_runner.py`
- `tests/test_ai_3d_generation.py`
- `tests/test_external_consent_gate.py`
- `tests/test_remote_provider_mock.py`
- `tests/test_ai_provider_security.py`

---

## 2. Preflight

```
GET /api/ai-3d/preflight → ok: true
execution_mode: wsl_subprocess
distro: Ubuntu-24.04
wsl_exe: ok (C:\Windows\system32\wsl.exe)
python: ok (/home/lenovo/sf3d_venv/bin/python)
worker_script: ok
dry_run_contract: ok (correctly fails on nonexistent image)
```

---

## 3. Default API E2E — Options Omitted

**Method:** POST `/api/ai-3d/process/{session_id}` with body `{"options": {}}` — no quality_mode, no background_removal_enabled.

**Session:** `ai3d_000e4bbdf1b6`

**Manifest assertions verified:**

| Field | Expected | Actual | Pass |
|---|---|---|---|
| `quality_mode` | `"high"` | `"high"` | ✓ |
| `resolved_quality.input_size` | `1024` | `1024` | ✓ |
| `resolved_quality.texture_resolution` | `1024` | `1024` | ✓ |
| `preprocessing.background_removed` | `true` | `true` | ✓ |
| `preprocessing.mask_source` | `"rembg"` | `"rembg"` | ✓ |
| `preprocessing.input_size` | `1024` | `1024` | ✓ |
| `worker_metadata.input_size` | `1024` | `1024` | ✓ |
| `worker_metadata.device` | `"cuda"` | `"cuda"` | ✓ |
| `provider_status` | `"ok"` | `"ok"` | ✓ |
| `output_glb_path` | exists | `data\ai_3d\ai3d_000e4bbdf1b6\derived\output.glb` | ✓ |
| `output_size_bytes` | > 0 | `1,008,076` (~0.96 MB) | ✓ |
| `duration_sec` | — | `24.81s` | — |
| `peak_mem_mb` | — | `6173.0 MB` | — |

**Conclusion:** When no options are provided, the API runtime defaults to `quality_mode=high` and `background_removal_enabled=true` as set in Phase 3C.

---

## 4. Override API E2E — Explicit balanced + bg=false

**Method:** POST `/api/ai-3d/process/{session_id}` with body:
```json
{"options": {"quality_mode": "balanced", "background_removal_enabled": false}}
```

**Session:** `ai3d_400c33171608`

**Manifest assertions verified:**

| Field | Expected | Actual | Pass |
|---|---|---|---|
| `quality_mode` | `"balanced"` | `"balanced"` | ✓ |
| `resolved_quality.input_size` | `768` | `768` | ✓ |
| `resolved_quality.max_candidates` | `3` | `3` | ✓ |
| `preprocessing.background_removed` | `false` | `false` | ✓ |
| `preprocessing.mask_source` | not `"rembg"` | `"fallback_center_crop"` | ✓ |
| `preprocessing.input_size` | `768` | `768` | ✓ |
| `worker_metadata.input_size` | `768` | `768` | ✓ |
| `worker_metadata.device` | `"cuda"` | `"cuda"` | ✓ |
| `provider_status` | `"ok"` | `"ok"` | ✓ |
| `output_glb_path` | exists | `data\ai_3d\ai3d_400c33171608\derived\output.glb` | ✓ |
| `output_size_bytes` | > 0 | `1,020,504` (~0.97 MB) | ✓ |
| `duration_sec` | — | `18.47s` | — |

**Conclusion:** Per-request overrides work correctly. Providing explicit options overrides the defaults without affecting other sessions.

---

## 5. UI Static Verification

File: `ui/ai_3d_studio.html`

| Check | Expected | Result |
|---|---|---|
| `quality-select` element exists | yes | ✓ (line 94) |
| `high` option has `selected` attribute | yes | ✓ (`<option value="high" selected>`) |
| `bg-removal-checkbox` exists with `checked` | yes | ✓ (`<input type="checkbox" id="bg-removal-checkbox" checked ...>`) |
| `ultra-warning` element exists | yes | ✓ (hidden by default, shown via `onQualityChange()` when ultra selected) |
| Ultra warning text correct | "Uses more VRAM..." | ✓ ("Ultra mode: Uses more VRAM, is slower, and creates larger files. Use only for texture-critical assets.") |
| `processSession` sends `options.quality_mode` | yes | ✓ (line 319) |
| `processSession` sends `options.background_removal_enabled` | yes | ✓ (line 320) |
| `processSession` sends `options.external_provider_consent` | yes | ✓ (line 318) |

---

## 6. Output GLB Comparison

| Run | quality_mode | bg | GLB size | input_size | duration |
|---|---|---|---|---|---|
| Default (high + bg=on) | high | on | 0.96 MB | 1024 | 24.81s |
| Override (balanced + bg=off) | balanced | off | 0.97 MB | 768 | 18.47s |

Note: Slightly similar GLB sizes because both runs used the same synthetic image. The `high` run used rembg preprocessing which correctly isolated the foreground before SF3D inference.

---

## 7. Confirmations

**Default runtime is high + background removal on.**
When no `quality_mode` or `background_removal_enabled` options are provided in the API request, the pipeline resolves to `quality_mode=high` with `background_removal_enabled=true`, using `input_size=1024` and rembg foreground isolation. This is confirmed by live API E2E run `ai3d_000e4bbdf1b6`.

**External providers were not touched.**
`ai_3d_remote_providers_enabled` remains `False` by default. `meshy_enabled` and `rodin_enabled` remain `False`. No external provider code was modified in Phase 3C or 3D. The test `test_remote_providers_disabled_by_default` and `test_remote_providers_still_disabled_by_default` (Phase 3C) both pass.
