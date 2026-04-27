# Phase 6.1: SAM2 Integration — Hardened DEV-SUBSET Plan

This plan outlines the integration of Segment Anything Model v2 (SAM2) for semantic segmentation, targeting a development subset for validation before any production use.

## ⚠️ Critical Safety Constraints

1. **Default is always `legacy`.**  `SEGMENTATION_METHOD=legacy` is the default in settings and `.env`.
2. **SAM2 must NEVER become the production default.**  It requires explicit `SAM2_ENABLED=true`.
3. **Do NOT commit model weights** (`models/sam2/*.pt`) to the repository.
4. **torch and segment-anything-2 are NOT hard dependencies.**  They are optional and only imported when `SAM2_ENABLED=true`.  Normal `pip install` of this project does not require them.
5. **Real SAM2 inference is NOT currently active.**  `sam2_backend.py` raises `NotImplementedError`.  `segment_video()` returns `{}`.  This is intentional.
6. **User review is required** before implementing real SAM2 inference.

## 1. Env Flags (Phase 6.1)

| Flag | Default | Description |
|------|---------|-------------|
| `SEGMENTATION_METHOD` | `legacy` | Which segmentation path to request: `legacy` or `sam2` |
| `SAM2_ENABLED` | `false` | Master kill-switch. Must be `true` to import torch/SAM2 |
| `SAM2_DEVICE` | `cuda` | torch device for SAM2 inference |
| `SAM2_MODEL_CFG` | `sam2_hiera_l.yaml` | SAM2 model config YAML |
| `SAM2_CHECKPOINT` | `models/sam2/sam2_hiera_large.pt` | Path to model weights |
| `SAM2_FALLBACK_TO_LEGACY` | `true` | Fall back silently on SAM2 failure |
| `SAM2_REVIEW_ONLY` | `true` | SAM2-produced assets marked review_only |
| `SAM2_PROMPT_MODE` | `center_box` | Prompt strategy: `center_box`, `center_point`, `auto` |
| `SAM2_MAX_FRAMES` | `0` | Max frames for SAM2 propagation (0 = unlimited) |

## 2. Architectural Changes

### A. `modules/ai_segmentation/sam2_wrapper.py`
- Reads all SAM2 env/settings values from `settings.py` singleton.
- Reports full status via `get_status()`:
  `sam2_enabled`, `sam2_available`, `sam2_model_loaded`, `sam2_inference_ran`,
  `sam2_error_reason`, `device`, `checkpoint_exists`, `model_cfg`, `checkpoint`.
- If `SAM2_ENABLED=false`, does NOT attempt torch/SAM2 import.
- If checkpoint missing, reports exact reason.
- `segment_video()` returns `None` when unavailable, `{}` when stub runs.

### B. `modules/capture_workflow/segmentation_backends/sam2_backend.py`
- Raises `NotImplementedError` (intentional stub).
- ObjectMasker catches this and falls back to heuristic backend.

### C. `modules/capture_workflow/object_masker.py`
- Checks `SAM2_ENABLED` kill-switch before importing SAM2Wrapper.
- Metadata always includes:
  - `segmentation_method` — actual backend used
  - `requested_segmentation_method` — what was requested (only if fallback)
  - `fallback_used` — bool
  - `fallback_reason` — human-readable reason

### D. `modules/ai_segmentation/segmentation_factory.py`
- Uses `settings.segmentation_method` (not raw `os.getenv`).
- Checks `SAM2_ENABLED` kill-switch.

## 3. Oracle Experiments (NOT Real SAM2)

The following scripts use ground-truth masks to simulate perfect segmentation. They are NOT real SAM2 inference:

- `scripts/simulate_oracle_mask_experiment.py` — Copies GT masks as "oracle" masks
- `scripts/run_oracle_cleanup_experiment.py` — Runs cleanup with oracle masks

Any `segmentation_method: sam2` metadata in these experiments is synthetic.

## 4. Test Coverage

Tests in `tests/modules/capture_workflow/test_object_masker_sam2_flag.py`:
- Default env uses legacy
- `SEGMENTATION_METHOD=sam2 + SAM2_ENABLED=false` → falls back to legacy
- `SEGMENTATION_METHOD=sam2 + SAM2 unavailable` → falls back to legacy
- SAM2 wrapper status reports checkpoint missing
- SAM2 backend `NotImplementedError` does not crash ObjectMasker
- SAM2-used assets are review_only / delivery_ready=false

## 5. What Is NOT Implemented

- **Depth Anything** — Not in scope. Segmentation is the bottleneck.
- **Real SAM2 inference** — `build_sam2_video_predictor` call is commented out.
- **SAM2 as production default** — Explicitly prohibited.

---

## 🛑 User Review Required
**Real SAM2 inference implementation requires explicit user approval.  The current code is hardened scaffolding only.**
