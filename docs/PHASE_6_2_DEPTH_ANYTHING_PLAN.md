# Phase 6.2: Depth Anything — Gated Scaffold Plan

## ⚠️ Critical Safety Constraints

1. **DEPTH_ANYTHING_ENABLED defaults to `false`.** Not active in production.
2. **Depth Anything is NOT a replacement for segmentation.** Segmentation is the bottleneck.
3. **Depth prior is ONLY allowed after SAM2/segmentation quality passes thresholds:**
   - Segmentation IoU ≥ 0.85
   - Leakage ratio ≤ 0.05
   - Mask confidence ≥ 0.75
4. **No production use without user review.**
5. **No model weights committed to the repo.**
6. **torch, transformers, depth-anything-v2 are NOT hard dependencies.**
7. **All depth outputs default to `review_only=true`.**

## Decision: When Can Depth Anything Be Considered?

```
SAM2 segmentation produces clean masks
    └─ IoU ≥ 0.85 against ground truth
    └─ Leakage ≤ 5%
    └─ Mask confidence ≥ 0.75
        └─ THEN depth prior MAY be considered
        └─ Still requires DEPTH_ANYTHING_ENABLED=true
        └─ Still requires valid checkpoint on disk
        └─ Still requires user review approval
```

If SAM2 does NOT improve segmentation, do NOT proceed to Depth Anything.

## Env Flags

| Flag | Default | Description |
|------|---------|-------------|
| `DEPTH_ANYTHING_ENABLED` | `false` | Master kill-switch |
| `DEPTH_ANYTHING_DEVICE` | `cuda` | torch device |
| `DEPTH_ANYTHING_MODEL` | `depth-anything-v2-small` | Model variant |
| `DEPTH_ANYTHING_CHECKPOINT` | `models/depth_anything/...` | Weights path |
| `DEPTH_ANYTHING_FALLBACK_TO_NONE` | `true` | Silent fallback |
| `DEPTH_ANYTHING_REVIEW_ONLY` | `true` | Outputs flagged review-only |
| `DEPTH_ANYTHING_MAX_FRAMES` | `0` | Frame limit (0=unlimited) |
| `DEPTH_PRIOR_MIN_SEGMENTATION_IOU` | `0.85` | Min IoU to allow depth |
| `DEPTH_PRIOR_MAX_LEAKAGE_RATIO` | `0.05` | Max leakage to allow depth |
| `DEPTH_PRIOR_MIN_MASK_CONFIDENCE` | `0.75` | Min confidence for depth |

## Module Structure (scaffold)

```
modules/ai_depth/
  __init__.py
  depth_anything_wrapper.py    # Gated wrapper, same pattern as SAM2
  depth_prior_policy.py        # Decision rules + coverage classification
```

## Coverage / Completion Policy

| Observed Surface | Status | AI Completion |
|-----------------|--------|---------------|
| ≥ 70% | `production_candidate` | Not needed |
| 50–70% | `review_ready` | Allowed if enabled |
| 30–50% | `preview_only` | Allowed if enabled |
| < 30% | `failed` | Not allowed |

Critical regions (logo, label, text, brand marks) must NEVER be hallucinated.
Synthesized geometry/texture must be marked in metadata.

## What Is NOT Implemented
- Real Depth Anything inference
- Integration into reconstruction pipeline
- Generative surface completion
- AI as default path

---
🛑 **User Review Required** before implementing real Depth Anything inference or enabling it in reconstruction.
