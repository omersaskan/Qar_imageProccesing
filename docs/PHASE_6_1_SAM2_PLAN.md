# Phase 6.1: SAM2 Integration — Hardened DEV-SUBSET Plan

## ⚠️ Safety Constraints

1. **Default is always `legacy`.** SAM2 is never the production default.
2. **SAM2_ENABLED must be explicitly `true`** for torch/sam2 import.
3. **Do NOT commit model weights** to the repository.
4. **torch/sam2 are NOT hard deps.** Normal install doesn't need them.
5. **User review required** before enabling SAM2 in live reconstruction.

## Env Flags

| Flag | Default | Description |
|------|---------|-------------|
| `SEGMENTATION_METHOD` | `legacy` | `legacy` or `sam2` |
| `SAM2_ENABLED` | `false` | Master kill-switch |
| `SAM2_DEVICE` | `cuda` | torch device |
| `SAM2_MODEL_CFG` | `sam2_hiera_l.yaml` | Model config |
| `SAM2_CHECKPOINT` | `models/sam2/sam2_hiera_large.pt` | Weights path |
| `SAM2_FALLBACK_TO_LEGACY` | `true` | Silent fallback |
| `SAM2_REVIEW_ONLY` | `true` | Assets flagged review-only |
| `SAM2_PROMPT_MODE` | `center_box` | Prompt strategy |
| `SAM2_MAX_FRAMES` | `0` | Propagation limit |

## Architecture

### Real Inference Path (gated)
When SAM2_ENABLED=true + checkpoint exists + torch+sam2 installed:
1. `SAM2Wrapper.__init__()` calls `build_sam2_video_predictor()`
2. `SAM2Backend.segment()` calls `SAM2Wrapper.segment_frame()`
3. `prompting.generate_prompts()` creates point/box prompts
4. Real mask is returned with full metadata

### Fallback Chain
If any condition fails, ObjectMasker falls back to heuristic backend
with metadata: `requested_segmentation_method`, `fallback_used`, `fallback_reason`.

### Prompt Strategies (prompting.py)
- `center_point`: frame center or legacy mask centroid
- `center_box`: legacy mask bbox or center 50% of frame
- `auto`: prefers legacy bbox if confidence > 0.40

## Oracle Experiments (NOT Real SAM2)
- `scripts/simulate_oracle_mask_experiment.py`
- `scripts/run_oracle_cleanup_experiment.py`

These use GT masks, NOT real SAM2 inference.

## Evaluation
`scripts/run_sam2_dev_subset.py` — compares legacy vs SAM2 with IoU/leakage.

---
🛑 **User Review Required** before enabling real SAM2 in live jobs.
