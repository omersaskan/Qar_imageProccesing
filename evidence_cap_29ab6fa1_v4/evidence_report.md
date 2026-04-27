# Reconstruction Evidence Report: cap_29ab6fa1
**Status:** ❌ FAILED
**Exported At:** 2026-04-27T14:18:56.326044Z

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
- DensifyPointCloud-260425111902005BE5.log
- DensifyPointCloud-260425111902006625.log
- DensifyPointCloud-2604251119030052B9.log
- DensifyPointCloud-260425111903005805.log
- DensifyPointCloud-260425111942000FF1.log
- DensifyPointCloud-260425111942002C69.log
- DensifyPointCloud-2604251119420058FD.log
- DensifyPointCloud-260425111942006015.log
- DensifyPointCloud-2604251129150068BD.log
- DensifyPointCloud-260425112915006A09.log
- DensifyPointCloud-260425112915006B39.log
- DensifyPointCloud-2604251129160046B9.log
- DensifyPointCloud-260425113003001935.log
- DensifyPointCloud-260425113003003955.log
- DensifyPointCloud-260425113003004AF5.log
- DensifyPointCloud-260425113004002B1D.log
- DensifyPointCloud-2604251131540057AD.log
- DensifyPointCloud-26042511315400663D.log
- DensifyPointCloud-260425113155006021.log
- DensifyPointCloud-260425113155006E05.log
- DensifyPointCloud-260426065528002209.log
- InterfaceCOLMAP-2604251119020022A1.log
- InterfaceCOLMAP-260425111902004239.log
- InterfaceCOLMAP-260425111903001FE1.log
- InterfaceCOLMAP-260425111903002A3D.log
- InterfaceCOLMAP-260425111941004EBD.log
- InterfaceCOLMAP-260425111942004A9D.log
- InterfaceCOLMAP-260425111942006339.log
- InterfaceCOLMAP-260425111942006DDD.log
- InterfaceCOLMAP-260425112915001EE9.log
- InterfaceCOLMAP-260425112915002109.log
- InterfaceCOLMAP-2604251129150063F9.log
- InterfaceCOLMAP-26042511291500657D.log
- InterfaceCOLMAP-260425113003000CA5.log
- InterfaceCOLMAP-260425113003004A91.log
- InterfaceCOLMAP-260425113003005595.log
- InterfaceCOLMAP-260425113004005C89.log
- InterfaceCOLMAP-2604251131540019E1.log
- InterfaceCOLMAP-2604251131540060A1.log
- InterfaceCOLMAP-260425113155002595.log
- InterfaceCOLMAP-260425113155002739.log
- InterfaceCOLMAP-260426065533001F09.log
- reconstruction.log
- ReconstructMesh-260425111902005155.log
- ReconstructMesh-260425111902006741.log
- ReconstructMesh-26042511190300151D.log
- ReconstructMesh-260425111903006855.log
- ReconstructMesh-260425111942000DE1.log
- ReconstructMesh-2604251119420022E1.log
- ReconstructMesh-260425111942003B5D.log
- ReconstructMesh-260425111942005681.log
- ReconstructMesh-260425112915002E29.log
- ReconstructMesh-260425112915005335.log
- ReconstructMesh-260425112915006D2D.log
- ReconstructMesh-260425112916004875.log
- ReconstructMesh-2604251130030028D1.log
- ReconstructMesh-260425113003005CB5.log
- ReconstructMesh-2604251130030060F5.log
- ReconstructMesh-260425113004001CFD.log
- ReconstructMesh-260425113154002A6D.log
- ReconstructMesh-260425113154005169.log
- ReconstructMesh-2604251131550019DD.log
- ReconstructMesh-260425113155004659.log
- TextureMesh-26042511190200207D.log
- TextureMesh-260425111902006D1D.log
- TextureMesh-26042511190300357D.log
- TextureMesh-2604251119030067BD.log
- TextureMesh-2604251119410059E5.log
- TextureMesh-260425111942000245.log
- TextureMesh-2604251119420021B5.log
- TextureMesh-2604251119420041BD.log
- TextureMesh-260425112915001E41.log
- TextureMesh-2604251129150054E1.log
- TextureMesh-26042511291500675D.log
- TextureMesh-260425112916004755.log
- TextureMesh-260425113003005069.log
- TextureMesh-2604251130030054F1.log
- TextureMesh-2604251130030062DD.log
- TextureMesh-260425113004002F61.log
- TextureMesh-2604251131540016D5.log
- TextureMesh-260425113154005F11.log
- TextureMesh-260425113155002329.log
- TextureMesh-260425113155006C4D.log

## Artifacts
See `artifact_tree.txt` for full hierarchy.