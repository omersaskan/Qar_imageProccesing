# AI 3D Phase 4B.2 — Mesh Stats & AR Readiness E2E Verification Report

- **Date**: 2026-05-06 02:29 UTC
- **Commit SHA**: `522f2c8f3af4233fec90f0ce4dc89e582cb3d65b`
- **Branch**: main (clean, in sync with origin/main)
- **Environment**: Local Windows 11 / WSL2 Ubuntu-24.04 / SF3D wsl_subprocess mode

---

## 1. Test Suite

```
py -m pytest tests/test_ai_3d_generation.py tests/test_phase3a_benchmark_runner.py \
             tests/test_quality_profiles.py tests/test_external_consent_gate.py \
             tests/test_remote_provider_mock.py tests/test_ai_provider_security.py -q

217 passed in 11.68s
```

---

## 2. API E2E — Default Flow

**Input**: `scratch/ai3d_benchmark_inputs/simple_center_object.png`  
**Options sent**: `{}` (empty — all defaults used)  
**Session**: `ai3d_9ddf87bb168d`

### Resolved defaults (confirmed in manifest)

| Field | Value |
| :--- | :--- |
| `quality_mode` | `high` |
| `input_size` | 1024 |
| `texture_resolution` | 1024 |
| `background_removal_enabled` | `true` (rembg, bbox from alpha) |

### Core outcome

| Field | Value |
| :--- | :--- |
| `provider_status` | `ok` |
| `status` | `review` |
| `output_glb_path` | `data\ai_3d\ai3d_9ddf87bb168d\derived\output.glb` |
| `missing_outputs` | `[]` |
| `external_provider` | `false` |
| `duration_sec` | `26.38` |
| `output_size_bytes` | `1 007 268` |

---

## 3. `mesh_stats` Block (from manifest)

```json
{
  "enabled": true,
  "available": true,
  "vertex_count": 15357,
  "face_count": 26828,
  "geometry_count": 1,
  "error": null
}
```

Top-level promotion also confirmed:

```json
"vertex_count": 15357,
"face_count": 26828
```

---

## 4. `ar_readiness` Block (from manifest)

```json
{
  "enabled": true,
  "score": 95,
  "verdict": "mobile_ready",
  "checks": {
    "glb_exists":         { "ok": true,  "value": "data\\...\\output.glb" },
    "file_size_mb":       { "ok": true,  "value": 0.961 },
    "vertex_count":       { "ok": true,  "value": 15357 },
    "face_count":         { "ok": true,  "value": 26828 },
    "texture_resolution": { "ok": true,  "value": 1024 },
    "provider_status":    { "ok": true,  "value": "ok" },
    "quality_gate":       { "ok": true,  "value": "review" }
  },
  "warnings": [],
  "recommendations": []
}
```

**Mesh stats → AR readiness linkage confirmed:**
- `ar_readiness.checks.vertex_count.value` (15 357) == `mesh_stats.vertex_count` ✓
- `ar_readiness.checks.face_count.value` (26 828) == `mesh_stats.face_count` ✓

---

## 5. Postprocessing (no GLB modification)

```json
"postprocessing": {
  "enabled": true,
  "normalize": { "applied": false, "reason": "not_implemented_yet" },
  "optimize":  { "applied": false, "reason": "optimizer_not_configured" },
  "validate":  { "applied": false, "reason": "validator_not_configured" }
}
```

> **Confirmed: Mesh stats are read-only and do not modify GLB.**  
> No optimizer or validator ran. The GLB produced by SF3D was delivered as-is.

---

## 6. Benchmark Row (`--modes high --bg-modes on --limit 1`)

Input: `noisy_background_object.png`

| Field | Value |
| :--- | :--- |
| `vertex_count` | 21 881 |
| `face_count` | 39 144 |
| `mesh_stats_available` | `true` |
| `ar_score` | `95` |
| `ar_verdict` | `mobile_ready` |

Benchmark sourced vertex/face counts from `manifest["mesh_stats"]` (pre-computed by pipeline) without duplicating the trimesh call.

---

## 7. Confirmations

| Check | Result |
| :--- | :--- |
| Mesh stats are read-only — GLB not modified | **CONFIRMED** |
| External providers (Meshy/Rodin/Tripo/Hunyuan) not touched | **CONFIRMED** — `external_provider: false` in all manifests |
| SF3D provider core unchanged | **CONFIRMED** |
| Quality defaults unchanged (`high` + bg-on) | **CONFIRMED** |
| trimesh installed and functional | **CONFIRMED** — `mesh_stats.available: true` |

---

## 8. Phase 4B.1 Integration Summary

```
modules/ai_3d_generation/mesh_stats.py   — shared trimesh extractor
modules/ai_3d_generation/pipeline.py     — step 7.5: extract + store in manifest
modules/ai_3d_generation/ar_readiness.py — reads from mesh_stats as fallback
scripts/run_ai3d_benchmark.py            — prefers manifest mesh_stats, no duplicate trimesh call
tests/test_ai_3d_generation.py           — 10 new tests (217 total passing)
```

Phase 4B.1 is **CLOSED**. AR readiness now uses real mesh geometry counts from every SF3D run.
