# SF3D WSL2 Provider Runtime Integration Report — Phase 4D

**Date:** 2026-05-03  
**Status:** IMPLEMENTED ✅  
**Prior phase:** Phase 4C WSL2 Smoke Test — PASS (output.glb 1.3 MB, RTX 5060 SM_12.0)

---

## 1. Settings

All new settings live in `modules/operations/settings.py` and can be overridden via environment variables.

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `sf3d_execution_mode` | `SF3D_EXECUTION_MODE` | `disabled` | `disabled` \| `local_windows` \| `wsl_subprocess` \| `remote_http` |
| `sf3d_wsl_distro` | `SF3D_WSL_DISTRO` | `Ubuntu-24.04` | WSL2 distro name passed to `wsl.exe -d` |
| `sf3d_wsl_python_path` | `SF3D_WSL_PYTHON_PATH` | `/home/lenovo/sf3d_venv/bin/python` | Python interpreter inside WSL2 venv |
| `sf3d_wsl_repo_root` | `SF3D_WSL_REPO_ROOT` | `/mnt/c/Users/Lenovo/.gemini/antigravity/scratch/Qar_imageProccesing` | WSL2 path to project root (for resolving `scripts/sf3d_worker.py`) |
| `sf3d_wsl_timeout_sec` | `SF3D_WSL_TIMEOUT_SEC` | `600` | Subprocess timeout for wsl.exe call (seconds) |
| `sf3d_wsl_output_copy_enabled` | `SF3D_WSL_OUTPUT_COPY_ENABLED` | `true` | (Reserved; output is written to Windows FS automatically via /mnt/ paths) |

**All pre-existing `SF3D_*` settings remain unchanged. Default `SF3D_ENABLED=false` and `SF3D_EXECUTION_MODE=disabled` — feature is opt-in.**

### To enable WSL2 mode in `.env`
```env
SF3D_ENABLED=true
SF3D_EXECUTION_MODE=wsl_subprocess
# Optional overrides (defaults shown):
# SF3D_WSL_DISTRO=Ubuntu-24.04
# SF3D_WSL_PYTHON_PATH=/home/lenovo/sf3d_venv/bin/python
# SF3D_WSL_REPO_ROOT=/mnt/c/Users/Lenovo/.gemini/...
# SF3D_WSL_TIMEOUT_SEC=600
```

---

## 2. Execution Mode Routing

`SF3DProvider.generate()` dispatches based on `sf3d_execution_mode`:

```
sf3d_execution_mode
├── disabled         → unavailable (error_code: sf3d_execution_mode_disabled)
├── local_windows    → _generate_local_windows()  [original behaviour]
├── wsl_subprocess   → _generate_wsl_subprocess()  [Phase 4D]
└── remote_http      → unavailable (not yet implemented)
```

`is_available()` validates mode-specific preconditions before dispatch:
- `local_windows`: `sf3d_python_path` exists on disk + `sf3d_worker_script` exists on disk
- `wsl_subprocess`: `sf3d_wsl_python_path` set + `sf3d_wsl_distro` set + `sf3d_wsl_repo_root` set + worker script reachable via Windows FS

---

## 3. Command Shape — `wsl_subprocess`

```python
[
  "wsl.exe",
  "-d", "Ubuntu-24.04",
  "--",
  "/home/lenovo/sf3d_venv/bin/python",
  "/mnt/c/.../scripts/sf3d_worker.py",
  "--image",              "/mnt/c/.../scratch/sf3d_smoke/input.png",
  "--output-dir",         "/mnt/c/.../data/ai_3d/<session>/derived",
  "--device",             "cuda",
  "--input-size",         "512",
  "--texture-resolution", "512",
  "--remesh",             "none",
  "--output-format",      "glb",
  # optional flags:
  "--no-remove-bg",       # if opts["no_remove_bg"]
  "--dry-run",            # if opts["dry_run"]
]
```

**Important:** `wsl.exe` is on `%PATH%` on Windows 11. No extra PATH setup needed.

---

## 4. Path Mapping

Windows paths are automatically converted to WSL2 `/mnt/X/...` format for the subprocess call, and converted back when processing the worker's JSON response.

| Direction | Function | Example |
|-----------|----------|---------|
| Windows → WSL2 | `_windows_to_wsl_path()` | `C:\Users\Foo\bar.png` → `/mnt/c/Users/Foo/bar.png` |
| WSL2 → Windows | `_wsl_to_windows_path()` | `/mnt/c/Users/Foo/out.glb` → `C:\Users\Foo\out.glb` |

Output paths that remain in WSL-only space (e.g., `/tmp/...`) would not be accessible from Windows — callers should always use Windows-side paths as `output_dir` to ensure bidirectional access via `/mnt/`.

---

## 5. JSON Contract

`scripts/sf3d_worker.py` writes exactly one JSON object to **stdout**. All logs go to **stderr**.

### Success (`status: ok`)
```json
{
  "status": "ok",
  "output_path": "/mnt/c/.../derived/output.glb",
  "output_format": "glb",
  "model_name": "stable-fast-3d",
  "preview_image_path": null,
  "warnings": ["ai_generated_not_true_scan"],
  "metadata": {
    "device": "cuda",
    "input_size": 512,
    "texture_resolution": 512,
    "remesh": "none",
    "pretrained_model": "stabilityai/stable-fast-3d",
    "foreground_ratio": 0.85,
    "peak_mem_mb": 6173.5,
    "output_size_bytes": 1346664
  }
}
```

The provider adds `"execution_mode": "wsl_subprocess"` to `metadata` before returning.  
The `output_path` is normalized from WSL `/mnt/c/...` → Windows `C:\...` by `_wsl_to_windows_path()`.

### Unavailable (`status: unavailable`)
```json
{
  "status": "unavailable",
  "error_code": "sf3d_package_missing" | "sf3d_model_auth_required",
  "message": "..."
}
```
Worker exits 0. Provider maps to `status: unavailable`.

### Error (`status: failed`)
```json
{
  "status": "failed",
  "error_code": "sf3d_inference_error",
  "message": "..."
}
```
Worker exits 1. Provider maps to `status: failed`.

---

## 6. API Response Shape

`POST /api/ai-3d/process/{session_id}` now returns additional fields:

```json
{
  "session_id": "ai3d_...",
  "status": "review" | "ok" | "failed" | "unavailable",
  "execution_mode": "wsl_subprocess",
  "provider_status": "ok",
  "output_glb_path": "C:\\...\\derived\\output.glb",
  "missing_outputs": [],
  "peak_mem_mb": 6173.5,
  "worker_metadata": {
    "device": "cuda",
    "texture_resolution": 512,
    "peak_mem_mb": 6173.5,
    "output_size_bytes": 1346664,
    "execution_mode": "wsl_subprocess"
  },
  "warnings": ["ai_generated_not_true_scan"],
  "errors": [],
  "manifest": { ... }
}
```

---

## 7. Manifest Schema Additions

`build_manifest()` now accepts and persists:

| Field | Type | Description |
|-------|------|-------------|
| `execution_mode` | `str` | Active execution mode at generation time |
| `worker_metadata` | `dict` | Full metadata dict from the worker JSON |
| `peak_mem_mb` | `float \| None` | Extracted from `worker_metadata.peak_mem_mb` |
| `provider_failure_reason` | `str \| None` | Human-readable failure reason when status ≠ ok |
| `missing_outputs` | `list[str]` | Keys of expected outputs that are absent |

All new fields are **optional with defaults** — existing callers and tests remain unaffected.

---

## 8. UI Changes

`ui/ai_3d_studio.html` updated:

- **Provider row**: shows `execution_mode` badge (e.g. `WSL SUBPROCESS`) when active
- **Output grid** (10 fields):
  - Provider, Execution Mode, Model, Device, Status,
    Is True Scan, Geo Confidence, Peak GPU Mem, Output Size, Review Required
- "AI generated, not true scan" warning badge persists in all states

---

## 9. Tests

`tests/test_ai_3d_generation.py` — 68 tests (was 50, +18):

### New test classes

| Class | Tests | Coverage |
|-------|-------|---------|
| `TestSF3DWSLPathMapping` | 5 | `_windows_to_wsl_path` / `_wsl_to_windows_path` — C: drive, spaces, posix passthrough |
| `TestSF3DProviderWSLMode` | 9 | WSL availability, command construction, ok/failed/invalid/unavailable JSON, path normalization |
| `TestManifestExecutionMode` | 4 | `execution_mode`, `worker_metadata`, `peak_mem_mb`, `provider_failure_reason` in manifest |

### Updated existing tests
- `TestSF3DProviderAvailability._provider_with_settings()` — adds `sf3d_execution_mode="local_windows"` to defaults
- `TestSF3DProviderGenerate._make_provider()` — adds `sf3d_execution_mode="local_windows"` to defaults

---

## 10. Manual Smoke Command

The Phase 4C smoke inference is reproducible at any time:

```powershell
# Ensure HF token is set (huggingface-cli login or HF_TOKEN env var)
wsl -d Ubuntu-24.04 -- /home/lenovo/sf3d_venv/bin/python `
    /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/Qar_imageProccesing/scripts/sf3d_worker.py `
    --image /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/Qar_imageProccesing/scratch/sf3d_smoke/input.png `
    --output-dir /tmp/sf3d_smoke `
    --device cuda `
    --texture-resolution 512 `
    --no-remove-bg
```

**Phase 4C result:** `status: ok`, GLB 1,346,664 bytes, peak GPU 6,173.5 MB,  
mesh 21,350 vertices / 42,696 faces.  
Artifact: `scratch/sf3d_smoke/output.glb` (Windows copy).

---

## 11. Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| HuggingFace gated model requires login | 401 on first use | `huggingface-cli login` in WSL2 once |
| numpy pinned to 1.26.4 (transformers compat) | `tifffile`/`opencv` pip warnings | Runtime harmless; do not upgrade gpytoolbox without re-pinning |
| `wsl.exe` required — Windows-only | No Linux/macOS support for WSL mode | Use `local_windows` or `remote_http` on other platforms |
| Model weights not in repo | First run downloads ~2 GB | Cached in `~/.cache/huggingface/` after first run |
| WSL2 cold start adds ~2s | Negligible vs inference time (~30-60s) | Acceptable |
| `remote_http` mode stub only | Not yet implemented | Future phase |

---

## 12. Next: Quality Benchmark Plan

1. **Real input images** — run smoke test with real product photos (not synthetic sphere)
2. **Mesh quality metrics** — vertex count, face count, texture resolution, UV coverage
3. **Timing breakdown** — model load vs. inference vs. export per image size
4. **Memory profiling** — peak VRAM by texture resolution (512 / 1024 / 2048)
5. **Background removal** — compare `--no-remove-bg` vs rembg session on product cutouts
6. **Foreground ratio sweep** — test `--foreground-ratio` 0.75 / 0.85 / 0.95 impact on geometry
