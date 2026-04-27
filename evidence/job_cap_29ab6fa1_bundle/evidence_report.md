# Reconstruction Evidence Report: job_cap_29ab6fa1
**Exported At:** 2026-04-27T13:48:39.898411

## Delivery Checklist Summary
| Criterion | Status | Value | Required |
|-----------|--------|-------|----------|
| Capture Status | ✅ | sufficient | PASS or WARN |
| Accepted Frames | ✅ | 27 | >= 15 |
| Dense Masks Match | ❌ | dim_match=54, file_match=None | Exact/Dimension Match |
| Fallback White Ratio | ✅ | 0.0 | < 0.3 |
| Registered Image Count | ✅ | 54 | >= 10 |
| Fused Point Count | ✅ | 4019287 | >= 5000 |
| Object Isolation Method | ❌ | None | mask_guided or hybrid_pc_mask |
| Isolation Confidence | ❌ | None | >= 0.7 |
| Texture Count | ✅ | 1 | > 0 |
| Material Count | ✅ | 1 | > 0 |
| Texcoord 0 Exists | ✅ | True | True |
| Texture Applied | ✅ | True | True |
| Glb Validation | ❌ | None | PASS or REVIEW |

> [!CAUTION]
> Asset FAILED one or more delivery criteria.

## Configuration Snapshot
```json
{
  "env": "local_dev",
  "recon_pipeline": "colmap_dense",
  "require_textured_output": true,
  "recon_mesh_budget_faces": 1000000,
  "recon_mobile_target_faces": 100000
}
```

## Available Logs
- attempt_0_default\reconstruction.log
- attempt_1_denser_frames\reconstruction.log

## Artifacts
See `artifact_tree.txt` for full hierarchy.