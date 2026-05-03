# AI 3D SF3D — End-to-End API Smoke & Benchmark Report

**Date:** 2026-05-04  
**Phase:** 4E (Hardening + E2E Smoke)  
**Status:** E2E PASS ✅  
**Prior Phase:** 4D WSL2 Runtime Integration — IMPLEMENTED ✅  

---

## 1. E2E Smoke Test — Full API Flow

### Environment

| Setting | Value |
|---------|-------|
| `SF3D_ENABLED` | `true` |
| `SF3D_EXECUTION_MODE` | `wsl_subprocess` |
| `AI_3D_GENERATION_ENABLED` | `true` |
| `AI_3D_DEFAULT_PROVIDER` | `sf3d` |
| Server | `uvicorn` on `localhost:8001` |
| GPU | RTX 5060 Laptop (SM_12.0, 8 GB VRAM) — via WSL2 |
| WSL2 distro | `Ubuntu-24.04` |
| PyTorch | `cu128` (Blackwell-compatible) |

### Preflight — `GET /api/ai-3d/preflight`

All 5 checks passed:

| Check | Status | Detail |
|-------|--------|--------|
| `wsl_exe` | ✅ ok | `C:\Windows\system32\wsl.exe` |
| `distro` | ✅ ok | `Ubuntu-24.04` responds to echo |
| `python` | ✅ ok | `/home/lenovo/sf3d_venv/bin/python` exists |
| `worker_script` | ✅ ok | `scripts/sf3d_worker.py` reachable via `/mnt/` |
| `dry_run_contract` | ✅ ok | Returns `status: failed` for missing image path |

### Upload — `POST /api/ai-3d/upload`

```
Input: scratch/sf3d_smoke/input.png (PNG, 512×512 synthetic sphere)
Response: {"session_id": "ai3d_a332cee0a47e", "status": "uploaded"}
```

### Process — `POST /api/ai-3d/process/{session_id}`

**Full response:**

```json
{
  "session_id": "ai3d_a332cee0a47e",
  "status": "review",
  "execution_mode": "wsl_subprocess",
  "provider_status": "ok",
  "output_glb_path": "C:\\...\\derived\\output.glb",
  "missing_outputs": [],
  "peak_mem_mb": 6173.5,
  "worker_metadata": {
    "device": "cuda",
    "input_size": 512,
    "texture_resolution": 512,
    "remesh": "none",
    "pretrained_model": "stabilityai/stable-fast-3d",
    "foreground_ratio": 0.85,
    "peak_mem_mb": 6173.5,
    "output_size_bytes": 1354992,
    "execution_mode": "wsl_subprocess"
  },
  "warnings": [
    "no_mask_or_bbox_using_center_crop",
    "ai_generated_not_true_scan",
    "generated_geometry_estimated",
    "review_required"
  ],
  "errors": []
}
```

**Manifest timing and path diagnostics:**

```json
"generation_started_at":  "2026-05-03T22:22:27.773720+00:00",
"generation_finished_at": "2026-05-03T22:22:57.007470+00:00",
"duration_sec": 29.24,
"output_size_bytes": 1354992,
"path_diagnostics": {
  "source_input_path":     "C:\\...\\input\\upload.png",
  "generation_input":      "C:\\...\\derived\\ai3d_input.png",
  "output_dir":            "C:\\...\\derived",
  "output_glb_path":       "C:\\...\\derived\\output.glb",
  "generation_input_wsl":  "/mnt/c/.../derived/ai3d_input.png",
  "output_dir_wsl":        "/mnt/c/.../derived"
}
```

### Status — `GET /api/ai-3d/status/{session_id}`

```
status: review  |  provider_status: ok
```

### Output — `GET /api/ai-3d/output/{session_id}`

```
Content-Type: model/gltf-binary
Size:         1,354,992 bytes (1.29 MB)
```

### Prepared Input — `GET /api/ai-3d/prepared-input/{session_id}`

```
Content-Type: image/png
Size:         103,338 bytes
```

### Acceptance Criteria — All Met

| Criterion | Expected | Actual | Pass |
|-----------|----------|--------|------|
| `provider_status` | `ok` | `ok` | ✅ |
| `status` | `review` or `ok` | `review` | ✅ |
| `output_glb_path` | populated | `C:\...\output.glb` | ✅ |
| `worker_metadata.device` | `cuda` | `cuda` | ✅ |
| `peak_mem_mb` | populated | `6173.5 MB` | ✅ |
| `output_size_bytes` | populated | `1,354,992` | ✅ |
| `duration_sec` | populated | `29.24s` | ✅ |
| `missing_outputs` | `[]` | `[]` | ✅ |
| `GET /output` | GLB binary | `model/gltf-binary` 1.3 MB | ✅ |
| `GET /prepared-input` | PNG | `image/png` 101 KB | ✅ |
| `GET /preflight` | all checks pass | 5/5 ✅ | ✅ |

---

## 2. Phase 4E Hardening — Summary

All 7 hardening items implemented:

| Item | Status |
|------|--------|
| Fix `_windows_to_wsl_path()` relative/UNC edge cases | ✅ |
| Fix pipeline absolute path normalization + missing_outputs passthrough | ✅ |
| WSL preflight method + `/api/ai-3d/preflight` endpoint | ✅ |
| Process-level GPU lock → 409 when job running | ✅ |
| Manifest runtime metrics (timing, size, path_diagnostics) | ✅ |
| UI model-viewer GLB preview (`<model-viewer>`) | ✅ |
| Noisy stdout parsing (`_parse_worker_stdout` + worker redirect) | ✅ |

**Security:** No HF_TOKEN appears in any manifest, API response, log output, or source file.

---

## 3. Noisy stdout Bug — Root Cause & Fix

### Symptom

```
After Remesh 9298 18592
{"status":"ok","output_path":"..."}
```

Native gpytoolbox/trimesh remesh operations write progress directly to stdout fd=1, contaminating the single-JSON contract between worker and provider.

### Fix 1 — Provider (`sf3d_provider.py`)

New `_parse_worker_stdout(stdout)` helper:
1. Try `json.loads(full_stdout)` — happy path.
2. Scan lines bottom-up for last `{…}` JSON line — dirty path.
3. On dirty-path success: append `"worker_stdout_had_extra_lines"` to `warnings` and `logs`.
4. On total failure: return `None` → `sf3d_worker_invalid_json` error.

### Fix 2 — Worker (`sf3d_worker.py`)

`contextlib.redirect_stdout(sys.stderr)` wraps both `model.run_image()` and `mesh.export()`:
```python
with torch.no_grad(), autocast_ctx, contextlib.redirect_stdout(sys.stderr):
    mesh, glob_dict = model.run_image(...)

with contextlib.redirect_stdout(sys.stderr):
    mesh.export(str(output_glb), include_normals=True)

# _out({...}) called on clean stdout — outside all redirect contexts
```

This fix is active even when `remesh="none"` (as a preventive measure).

---

## 4. Test Coverage

| Class | Tests | Area |
|-------|-------|------|
| `TestParseWorkerStdout` | 8 | Noisy stdout parsing — happy/dirty/failure paths + full pipeline mock |
| `TestSF3DPathEdgeCases` | 5 | UNC, bare /mnt/c, lowercase drive, empty string |
| `TestSF3DBusyLock` | 3 | GPU lock held → busy, busy in known statuses, quality gate mapping |
| `TestSF3DPreflight` | 3 | Disabled mode, wsl.exe missing, structure check |
| `TestManifestTimingFields` | 5 | timing fields, output_size_bytes, path_diagnostics |

**Total:** 764 tests, 0 failures.

---

## 5. Benchmark — Run 1 (Smoke Sphere)

Only one image available for automated benchmark. Full 10-image benchmark to be scheduled as Phase 4F.

| # | Input | Size | Device | Texture | Duration | Peak VRAM | GLB Size | Mesh Quality |
|---|-------|------|--------|---------|----------|-----------|----------|--------------|
| 1 | Synthetic sphere (512×512 PNG) | 512 | cuda | 512 | 29.24s | 6173.5 MB | 1.29 MB | baseline |

---

## 6. Benchmark Plan — Phase 4F (10 Diverse Inputs)

### Methodology

For each input image, run:
```bash
# Upload
curl -X POST http://localhost:8001/api/ai-3d/upload \
  -F "file=@<image>" -F "provider=sf3d"

# Process
curl -X POST http://localhost:8001/api/ai-3d/process/<session_id>
```

Record from manifest:
- `duration_sec`
- `worker_metadata.peak_mem_mb`
- `worker_metadata.output_size_bytes`
- `quality_gate.verdict`
- `warnings`

### Input Matrix

| # | Subject | Background | Material | Expected Challenge |
|---|---------|------------|----------|--------------------|
| 1 | Metal key | White studio | Reflective metal | Shiny surface artifacts |
| 2 | Leather shoe | Natural | Matte + texture | Complex silhouette |
| 3 | White cup | Clean | Matte ceramic | Symmetric rotation |
| 4 | Toy car | White | Mixed | Small detail loss |
| 5 | Computer mouse | Dark | Matte plastic | Rounded uniform form |
| 6 | Fabric bag | Various | Soft/deformable | UV mapping on fabric |
| 7 | Matte cube | White | Flat matte | Edge sharpness |
| 8 | Black glossy object | Dark | Specular | Complete geometry loss |
| 9 | Product box | White | Flat with print | Text / label detail |
| 10 | Chair | Room | Wood/fabric | Large scale, legs |

### Scoring Rubric

| Metric | 1 (poor) | 3 (ok) | 5 (excellent) |
|--------|----------|--------|---------------|
| Silhouette accuracy | Major missing parts | Minor artifacts | Clean match |
| Surface completeness | >30% holes | <10% holes | No holes |
| Texture quality | Blurry/wrong | Acceptable | Sharp, correct |
| AR readiness | Unusable | Needs cleanup | Ship-ready |

---

## 7. Known Limitations

| Limitation | Observed | Impact |
|------------|----------|--------|
| `review_required=true` by policy | All runs → `status: review` | Expected — review gate is opt-in to disable |
| `no_mask_or_bbox_using_center_crop` | Input had no alpha/mask | Geometry may include background artifacts |
| Postprocess stubs | `normalize`, `optimize`, `validate` not yet implemented | GLB not gltf-transform-optimized |
| `remesh=none` default | No retopology | Mesh topology from SF3D directly |
| Single-image input only | No multi-view | AI estimation only — not photogrammetry |
| HuggingFace model cached | First run downloaded ~2 GB weights | Cached at `~/.cache/huggingface/` |

---

## 8. Reproduction Command

```powershell
# Start server
$env:SF3D_ENABLED="true"
$env:SF3D_EXECUTION_MODE="wsl_subprocess"
$env:AI_3D_GENERATION_ENABLED="true"
$env:AI_3D_DEFAULT_PROVIDER="sf3d"
py -m uvicorn modules.operations.api:app --host 0.0.0.0 --port 8001

# In another terminal:
# Preflight
Invoke-RestMethod http://localhost:8001/api/ai-3d/preflight

# Upload
$form = @{ file = Get-Item scratch\sf3d_smoke\input.png; provider = "sf3d" }
$up = Invoke-RestMethod -Uri http://localhost:8001/api/ai-3d/upload -Method POST -Form $form
$sid = $up.session_id

# Process (30-60s)
Invoke-RestMethod -Uri "http://localhost:8001/api/ai-3d/process/$sid" -Method POST -Body "{}" -ContentType application/json

# Download GLB
Invoke-WebRequest "http://localhost:8001/api/ai-3d/output/$sid" -OutFile output.glb
```
