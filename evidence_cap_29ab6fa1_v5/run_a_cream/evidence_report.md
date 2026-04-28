# Reconstruction Evidence Report: cap_29ab6fa1_v5_cream
**Status:** ❌ FAILED
**Exported At:** 2026-04-28T11:39:41.199697Z

## Delivery Checklist Summary
| Criterion | Status | Value | Required |
|-----------|--------|-------|----------|
| Capture Status | ❌ | None | PROD: PASS/sufficient, REVIEW: warn |
| Accepted Frames | ❌ | 0 | PROD: >= 30, REVIEW: >= 15 |
| Dense Masks Integrity | ❌ | exact=0, dim=0, total=0 | PROD/REVIEW: exact==total and dim==total |
| Fallback White Ratio | ❌ | 1.000 | PROD: <= 0.05, REVIEW: <= 0.1 |
| Registered Images | ❌ | count=0, ratio=0.00 | PROD: >= 20 & 70% |
| Fused Point Count | ❌ | 0 | PROD: >= 25000, REVIEW: >= 10000 |
| Object Isolation Method | ❌ | None | PROD: mask_guided/hybrid |
| Isolation Confidence | ❌ | 0.00 | PROD: >= 0.75, REVIEW: >= 0.6 |
| Texture Count | ❌ | 0 | > 0 |
| Material Count | ❌ | 0 | > 0 |
| Texcoord 0 Exists | ❌ | False | True |
| Texture Applied | ❌ | False | True |
| Glb Validation | ❌ | None | PROD: pass, REVIEW: review |

### ❌ Failure Reasons
- Capture status is None
- Too few frames: 0
- Dense mask mismatch or missing (exact=0/0)
- Excessive mask fallback: 1.000
- Low registration: 0 (0.00%)
- Insufficient density: 0 pts
- Unsupported isolation: None
- Low isolation confidence: 0.00
- Config requires texture but one or more texture components are missing.
- No textures found
- No materials found
- Missing UV coords
- Texture not applied to mesh
- GLB validation failed or missing: None

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
- **Dense Image Count:** 0
- **Dense Mask Count:** 0
- **Dense Mask Exact Matches:** 0
- **Dense Mask Dimension Matches:** 0
- **Dense Mask Fallback White Ratio:** 1.0
- **Registered Image Ratio:** 0.0

## Available Logs

## Artifacts
See `artifact_tree.txt` for full hierarchy.