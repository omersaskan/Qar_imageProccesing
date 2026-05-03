# SF3D Integration Plan

## Goal

Integrate Stable Fast 3D (SF3D) by Stability AI as the first self-hosted AI 3D generation provider,
running entirely in an isolated venv so the main application environment is never touched.

## Architecture Contract

| Concern | Decision |
|---|---|
| Process isolation | SF3D runs as a subprocess worker (`scripts/sf3d_worker.py`) |
| Python environment | Dedicated venv at `external/stable-fast-3d/.venv_sf3d` |
| Main env imports | `sf3d`, `torch`, and any ML package must **never** be imported in the main process |
| Config surface | All flags default to `False`/disabled; opt-in via env vars |
| Output contract | Worker writes exactly one JSON line to stdout; all logs go to stderr |

## Prerequisite: Installing SF3D

```bash
# 1. Clone the upstream repo
git clone https://github.com/Stability-AI/stable-fast-3d external/stable-fast-3d

# 2. Create an isolated venv
cd external/stable-fast-3d
python -m venv .venv_sf3d

# 3. Activate and install
.venv_sf3d\Scripts\activate      # Windows
pip install -r requirements.txt
pip install -e .
```

> **License**: Stable Fast 3D is released under the Stability AI Community License.
> Verify compliance before any commercial deployment.

## Configuration

Add to your `.env` (or set as environment variables):

```
# Enable AI 3D generation endpoints
AI_3D_GENERATION_ENABLED=true

# Enable the SF3D provider
SF3D_ENABLED=true

# Path to the SF3D venv Python executable
SF3D_PYTHON_PATH=external/stable-fast-3d/.venv_sf3d/Scripts/python.exe

# Optional tuning
SF3D_DEVICE=auto            # auto | cuda | cpu
SF3D_INPUT_SIZE=512
SF3D_TEXTURE_RESOLUTION=1024
SF3D_REMESH=none            # none | quad | triangle
SF3D_TIMEOUT_SEC=300

# Review policy (recommended: keep true)
SF3D_REQUIRE_REVIEW=true
AI_3D_REQUIRE_REVIEW=true
```

## Runtime Flow

```
POST /api/ai-3d/upload
  └─ stores input to data/ai_3d/{session_id}/input/

POST /api/ai-3d/process/{session_id}
  └─ generate_ai_3d()
       ├─ route_input()          → image / video
       ├─ select_best_frame()    → video only
       ├─ preprocess_input()     → ai3d_input.png (512×512 square)
       ├─ SF3DProvider.safe_generate()
       │    └─ subprocess: .venv_sf3d/python sf3d_worker.py
       │         prints JSON → {status, output_path, model_name, warnings, metadata}
       ├─ run_postprocess()      → stub (no-op today)
       ├─ quality_evaluate()     → ok / review / unavailable / failed
       └─ write_manifest()       → manifests/ai3d_manifest.json

GET /api/ai-3d/output/{session_id}        → serves output.glb
GET /api/ai-3d/prepared-input/{session_id} → serves ai3d_input.png
GET /api/ai-3d/status/{session_id}        → manifest JSON
```

## Unavailable Path (SF3D not installed)

When `sf3d_enabled=False` or the Python path does not exist, `SF3DProvider.is_available()`
returns `(False, reason)` and `safe_generate()` returns `{"status": "unavailable", ...}`.
The API endpoint surfaces `provider_failure_reason` in the response body.
The worker itself also handles `ImportError` on `import sf3d` and exits 0 with
`{"status": "unavailable", "error_code": "sf3d_package_missing"}`.

## Status Mapping

| Provider result | Quality gate | API status |
|---|---|---|
| `ok` + output exists + review required | `review` | 200 |
| `ok` + output exists + review not required | `ok` | 200 |
| `ok` + output missing | `failed` | 200 |
| `unavailable` | `unavailable` | 200 |
| `failed` | `failed` | 200 |

All 4xx/5xx errors are upload or server faults, not generation faults.

## Provenance Invariants (never relaxed)

Every AI 3D manifest must carry:

```json
"is_true_scan": false,
"geometry_confidence": "estimated",
"mode": "ai_generated_3d",
"asset_type": "ai_generated"
```

These fields distinguish AI-generated meshes from photogrammetry scans and must never
be set to values that imply real reconstruction.
