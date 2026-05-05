# AI 3D Phase 4B.2 — Mesh Stats Validation (1-Run)

- **Date**: 2026-05-06
- **Commit SHA**: `522f2c8` (feat: include mesh stats in AI 3D AR readiness)
- **Purpose**: Smoke-test that `mesh_stats` and `ar_readiness` integration works in a real SF3D run.
- **Scope**: 1 run — `high` quality + bg-on. NOT a full benchmark.

> This report is stored under `reports/ai3d_validation/phase4b2/` and does **not**
> overwrite canonical Phase 3B benchmark artifacts under `reports/ai3d_benchmark/`.

---

## Validation Run Result

| Field | Value |
| :--- | :--- |
| `input_filename` | `noisy_background_object.png` |
| `quality_mode` | `high` |
| `background_removal_enabled` | `true` |
| `status` | `review` |
| `provider_status` | `ok` |
| `vertex_count` | 21 881 |
| `face_count` | 39 144 |
| `mesh_stats_available` | `true` |
| `ar_score` | 95 |
| `ar_verdict` | `mobile_ready` |

## Confirmations

- `mesh_stats.available = true` — trimesh read real GLB geometry counts.
- `ar_readiness.checks.vertex_count.value` == `mesh_stats.vertex_count` ✓
- `ar_readiness.checks.face_count.value` == `mesh_stats.face_count` ✓
- GLB not modified — `postprocessing.optimize.applied = false`.
- External providers not touched — `external_provider = false`.
