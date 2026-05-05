# AI 3D Default Configuration

Benchmark reference: Phase 3B — 30/30 successful local SF3D runs across balanced/high/ultra × bg-off/on.
Commit: `9c4c766` · Report: `reports/ai3d_benchmark/AI_3D_PHASE3B_SF3D_FULL_BENCHMARK_REPORT.md`

---

## Recommended Configurations

| Use case | quality_mode | background_removal_enabled | Notes |
|---|---|---|---|
| **Default (production)** | `high` | `true` | Best balance of speed, quality, and VRAM |
| Fast preview | `balanced` | `true` | Faster (~18s), 768px input, same VRAM |
| Premium / manual retry | `ultra` | `true` | 2048px texture, +7s, +2 GB VRAM — use sparingly |

---

## Why high + bg-on is the default

From the Phase 3B benchmark (5 inputs × 3 modes × 2 bg states = 30 runs):

- `high` processes at 1024px input vs `balanced` at 768px, with **identical VRAM** (~6173 MB) and **faster** average runtime (17.0s vs 18.8s).
- `high` and `balanced` produce virtually identical GLB file sizes (~0.68 MB avg).
- Background removal (`rembg`) isolates the foreground object, reducing noise in the mesh — especially important for restaurant menu photos with complex backgrounds.
- `ultra` succeeded 10/10 but at significant cost: +7s, +2 GB VRAM, +38% file size. The additional texture resolution is not perceivable at typical mobile AR viewing distances.

---

## Settings (settings.py / environment variables)

| Setting | Default | Env var |
|---|---|---|
| `ai_3d_quality_mode` | `"high"` | `AI_3D_QUALITY_MODE` |
| `ai_3d_background_removal_enabled` | `True` | `AI_3D_BACKGROUND_REMOVAL_ENABLED` |

---

## External Providers

External providers (Meshy, Rodin, Tripo, Hunyuan, TRELLIS, SHARP) **remain disabled by default**.

| Setting | Default |
|---|---|
| `ai_3d_remote_providers_enabled` | `False` |
| `meshy_enabled` | `False` |
| `rodin_enabled` | `False` |

These settings were not changed by Phase 3C. The benchmark covered only local SF3D.

---

## Quality Profile Reference

| Mode | input_size | texture_resolution | max_candidates | Avg VRAM (MB) | Avg duration (s) | Avg GLB (MB) |
|---|---|---|---|---|---|---|
| balanced | 768  | 1024 | 3 | 6173 | 18.1 | 0.68 |
| high     | 1024 | 1024 | 5 | 6173 | 17.1 | 0.68 |
| ultra    | 1024 | 2048 | 8 | 8238 | 24.6 | 0.93 |

Ultra warning: Uses more VRAM, is slower, and creates larger files. Use only for texture-critical assets.
