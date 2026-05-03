# AI Generated 3D — Architecture

## Overview

The `modules/ai_3d_generation/` package implements a single-image-to-3D pipeline that is
architecturally isolated from both photogrammetry and depth-studio pipelines.

```
modules/ai_3d_generation/
  __init__.py          package marker
  provider_base.py     abstract base + safe_generate() contract
  sf3d_provider.py     SF3D subprocess provider
  input_preprocessor.py  image → canonical square input
  manifest.py          manifest schema + writer
  postprocess.py       GLB postprocessing stubs
  quality_gate.py      verdict assignment
  router.py            input-type/intent → pipeline decision

scripts/
  sf3d_worker.py       isolated worker (runs in SF3D venv)
```

## Package Boundaries

```
┌─────────────────────────────────────────────────────┐
│  modules/operations/api.py                          │
│    POST /api/ai-3d/upload                           │
│    POST /api/ai-3d/process/{id}  ──► generate_ai_3d │
│    GET  /api/ai-3d/output/{id}                      │
│    GET  /api/ai-3d/prepared-input/{id}              │
│    GET  /api/ai-3d/status/{id}                      │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  modules/ai_3d_generation/pipeline.py               │
│    generate_ai_3d()                                 │
│      1. copy input                                  │
│      2. route (image/video) + frame selection       │
│      3. preprocess_input() → ai3d_input.png         │
│      4. provider.safe_generate()                    │
│      5. run_postprocess()                           │
│      6. quality_evaluate()                          │
│      7. build_manifest() + write_manifest()         │
└──────────┬──────────────────────────────────────────┘
           │
┌──────────▼──────────┐    ┌───────────────────────────┐
│  SF3DProvider        │    │  AI3DProviderBase (ABC)   │
│  is_available()      │    │  safe_generate()          │
│  generate()          │    │  generate() [abstract]    │
│    subprocess call   │    │  is_available() [abstract]│
└──────────┬──────────┘    └───────────────────────────┘
           │  subprocess
┌──────────▼──────────────────────────────────────────┐
│  scripts/sf3d_worker.py  (SF3D venv Python)         │
│    argparse → dry-run / unavailable / inference     │
│    stdout: exactly one JSON object                  │
│    stderr: logs only                                │
└─────────────────────────────────────────────────────┘
```

## AI3DProviderBase Contract

Every provider must implement:

```python
def is_available(self) -> Tuple[bool, str]:
    """Return (available, reason_if_not)."""

def generate(self, input_image_path, output_dir, options=None) -> Dict:
    """
    Must return a dict with at minimum:
      status: "ok" | "unavailable" | "failed"
      output_path: str | None
      warnings: list[str]
      error: str | None
      error_code: str | None
    """
```

`safe_generate()` on the base class:
1. Calls `is_available()` → returns unavailable result if not ready
2. Calls `generate()` in a try/except → returns failed result on exception
3. Normalises non-standard status values to `"failed"`

No provider should ever raise from `safe_generate()`.

## Manifest Schema

```json
{
  "mode": "ai_generated_3d",
  "asset_type": "ai_generated",
  "is_true_scan": false,
  "geometry_confidence": "estimated",
  "source": "single_image_or_best_frame",
  "session_id": "...",
  "created_at": "...",
  "provider": "sf3d",
  "provider_status": "ok|unavailable|failed",
  "model_name": "stable-fast-3d",
  "license_note": "...",
  "source_input_path": "...",
  "input_type": "image|video",
  "selected_frame_path": null,
  "prepared_image_path": "...",
  "preprocessing": { ... },
  "postprocessing": { ... },
  "quality_gate": {
    "verdict": "ok|review|unavailable|failed",
    "output_exists": true,
    "warnings": ["ai_generated_not_true_scan", "generated_geometry_estimated"],
    "reason": null
  },
  "output_glb_path": "...",
  "output_format": "glb",
  "preview_image_path": null,
  "status": "ok|review|unavailable|failed",
  "review_required": true,
  "warnings": [...],
  "errors": [...]
}
```

## Quality Gate Verdicts

| Condition | Verdict |
|---|---|
| Provider status = `unavailable` | `unavailable` |
| Provider status ≠ `ok` | `failed` |
| Provider ok, output file missing | `failed` (reason: `output_glb_missing`) |
| Provider ok, output exists, `review_required=True` | `review` |
| Provider ok, output exists, `review_required=False` | `ok` |

The gate always appends `ai_generated_not_true_scan` and `generated_geometry_estimated`
to the warnings list — these are non-negotiable provenance signals.

## Router Logic

`decide_asset_pipeline(input_type, user_intent, capture_quality)`:

| Input | Intent | Quality | Pipeline |
|---|---|---|---|
| any | `debug` | any | `depth_studio` |
| `image` | any | any | `ai_generated_3d` |
| `video` | `advanced` | good/fair | `real_reconstruction` |
| `video` | default | good/fair | `ai_generated_3d` (fallback: real_reconstruction) |
| `video` | any | poor/None | `ai_generated_3d` |

## Isolation Invariants

1. `import sf3d` only ever appears in `scripts/sf3d_worker.py`.
2. `import torch` only ever appears in `scripts/sf3d_worker.py`.
3. The SF3D venv Python binary is invoked via `subprocess.run` only.
4. `SF3DProvider.generate()` checks `is_available()` again before building the command.
5. Worker exit 0 is permitted for `unavailable` — it is not a crash.
6. Worker exit 1 means a hard failure.

## Adding Future Providers

1. Create `modules/ai_3d_generation/<name>_provider.py` subclassing `AI3DProviderBase`.
2. Create `scripts/<name>_worker.py` with the same stdout-JSON contract.
3. Add a new `elif provider_name == "<name>"` branch in `pipeline._get_provider()`.
4. Add settings flags (`<name>_enabled`, `<name>_python_path`, etc.) to `settings.py`.
5. Add API checks for the new provider in `api.py` error reason logic.
