# Reconstruction Evidence Report: job_phase5_final_verification
**Status:** ✅ PRODUCTION READY
**Exported At:** 2026-04-27T14:46:52.016337Z

## Delivery Checklist Summary
| Criterion | Status | Value | Required |
|-----------|--------|-------|----------|
| Capture Status | ✅ | sufficient | PROD: PASS/sufficient, REVIEW: warn |
| Accepted Frames | ✅ | 40 | PROD: >= 30, REVIEW: >= 15 |
| Dense Masks Integrity | ✅ | exact=38, dim=38, total=38 | PROD/REVIEW: exact==total and dim==total |
| Fallback White Ratio | ✅ | 0.000 | PROD: <= 0.05, REVIEW: <= 0.1 |
| Registered Images | ✅ | count=38, ratio=0.95 | PROD: >= 20 & 70% |
| Fused Point Count | ✅ | 50000 | PROD: >= 25000, REVIEW: >= 10000 |
| Object Isolation Method | ✅ | mask_guided | PROD: mask_guided/hybrid |
| Isolation Confidence | ✅ | 0.88 | PROD: >= 0.75, REVIEW: >= 0.6 |
| Texture Count | ✅ | 1 | > 0 |
| Material Count | ✅ | 1 | > 0 |
| Texcoord 0 Exists | ✅ | True | True |
| Texture Applied | ✅ | True | True |
| Glb Validation | ✅ | pass | PROD: pass, REVIEW: review |

> [!TIP]
> Asset is PRODUCTION READY. Automated checks passed with high confidence.

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

## Detailed Metrics
- **Dense Image Count:** 38
- **Dense Mask Count:** 38
- **Dense Mask Exact Matches:** 38
- **Dense Mask Dimension Matches:** 38
- **Dense Mask Fallback White Ratio:** 0.0
- **Registered Image Ratio:** 0.95
- **Mask Support Ratio:** 0.92

## Available Logs
- reconstruction.log

## Artifacts
See `artifact_tree.txt` for full hierarchy.