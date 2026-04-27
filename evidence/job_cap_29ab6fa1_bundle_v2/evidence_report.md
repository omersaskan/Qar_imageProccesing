# Reconstruction Evidence Report: job_cap_29ab6fa1
**Status:** ❌ FAILED
**Exported At:** 2026-04-27T13:58:01.804819Z

## Delivery Checklist Summary
| Criterion | Status | Value | Required |
|-----------|--------|-------|----------|
| Capture Status | ✅ | sufficient | PROD: PASS/sufficient, REVIEW: warn |
| Accepted Frames | ⚠️ | 27 | PROD: >= 30, REVIEW: >= 15 |
| Dense Masks Integrity | ❌ | exact=0, dim=54, total=54 | PROD/REVIEW: exact==total and dim==total |
| Fallback White Ratio | ✅ | 0.000 | PROD: <= 0.05, REVIEW: <= 0.1 |
| Registered Images | ✅ | count=54, ratio=2.00 | PROD: >= 20 & 70% |
| Fused Point Count | ✅ | 4019287 | PROD: >= 25000, REVIEW: >= 10000 |
| Object Isolation Method | ❌ | None | PROD: mask_guided/hybrid |
| Isolation Confidence | ❌ | 0.00 | PROD: >= 0.75, REVIEW: >= 0.6 |
| Texture Count | ✅ | 1 | > 0 |
| Material Count | ✅ | 1 | > 0 |
| Texcoord 0 Exists | ✅ | True | True |
| Texture Applied | ✅ | True | True |
| Glb Validation | ❌ | None | PROD: pass, REVIEW: review |

### ❌ Failure Reasons
- Dense mask mismatch or missing (exact=0/54)
- Unsupported isolation: None
- Low isolation confidence: 0.00
- GLB validation failed: None

### ⚠️ Warning Reasons
- Low frame count: 27

> [!CAUTION]
> Asset FAILED delivery criteria. Do not deliver without remediation.

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
- **Dense Image Count:** 54
- **Dense Mask Count:** 54
- **Dense Mask Exact Matches:** 0
- **Dense Mask Dimension Matches:** 54
- **Dense Mask Fallback White Ratio:** 0.0
- **Registered Image Ratio:** 2.0

## Available Logs
- attempt_0_default\reconstruction.log
- attempt_1_denser_frames\reconstruction.log

## Artifacts
See `artifact_tree.txt` for full hierarchy.