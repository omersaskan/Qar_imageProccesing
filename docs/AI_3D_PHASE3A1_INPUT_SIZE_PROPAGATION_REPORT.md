# AI 3D Phase 3A.1 — Input Size Propagation Report

## Overview
This report documents the fix for the `input_size` mismatch between `resolved_quality` and `worker_metadata` in the AI 3D generation pipeline.

- **Commit SHA**: `3c2765c` — fix: propagate quality input size to SF3D worker
- **Date**: 2026-05-05
- **Test Result**: 203 passed

## Problem
Benchmark manifests showed a discrepancy:
- `resolved_quality.input_size` = 768 (balanced) / 1024 (high)
- `worker_metadata.input_size` = 512 (always)

The pipeline resolved input_size from quality profiles but did **not inject it** into `opts` before passing to the SF3D provider. The provider fell back to `settings.sf3d_input_size` (default: 512).

## Fix Applied

### `modules/ai_3d_generation/pipeline.py`
Added missing injection:
```diff
 # Inject resolved quality into options for provider
+opts["input_size"] = resolved_quality["input_size"]
 opts["texture_resolution"] = resolved_quality["texture_resolution"]
 opts["remesh"] = resolved_quality["remesh"]
```

### `modules/ai_3d_generation/sf3d_provider.py`
Added defensive clamping (64–1024) in both `_generate_local_windows` and `_generate_wsl_subprocess` before passing `--input-size` to the worker command.

### `scripts/sf3d_worker.py`
No changes needed — worker already reports `args.input_size` in metadata.

## Verification

### Before Fix
| Mode | resolved_quality.input_size | worker_metadata.input_size |
|:---|:---|:---|
| balanced | 768 | **512** (wrong) |
| high | 1024 | **512** (wrong) |

### After Fix
| Mode | resolved_quality.input_size | worker_metadata.input_size | Match |
|:---|:---|:---|:---|
| balanced | 768 | **768** | Yes |
| high | 1024 | **1024** | Yes |

The benchmark was re-run after the input_size propagation fix and confirmed:
balanced `resolved_quality.input_size=768` equals `worker_metadata.input_size=768`;
high `resolved_quality.input_size=1024` equals `worker_metadata.input_size=1024`.

### SF3D Accepted 1024
- SF3D accepted `input_size=1024` successfully.
- Both runs completed with `provider_status=ok` and valid GLB output.

## Notes
- External providers were not touched.
- This is a correctness fix only — no new features.
