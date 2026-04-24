# Dataset Schema

## 1. Session Training Manifest

Each capture session should produce a privacy-safe training manifest.

Recommended path:

```text
data/training_manifests/{session_id}.json
data/captures/{session_id}/reports/training_manifest.json
```

---

## 2. Example Manifest

```json
{
  "schema_version": "1.0",
  "session_id": "cap_xxx",
  "product_id_hash": "sha256_hash",
  "asset_id": "asset_xxx",
  "created_at": "2026-04-24T12:00:00Z",
  "consent_status": "unknown",
  "eligible_for_training": false,

  "device": {
    "platform": "ios|android|web|unknown",
    "model": "unknown",
    "has_lidar": false,
    "camera_resolution": "1920x1080",
    "fps": 30
  },

  "capture": {
    "mode": "guided_camera|admin_video_upload|imported_video",
    "duration_sec": 42,
    "original_frame_count": 84,
    "accepted_frame_count": 42,
    "capture_score": 82.5,
    "coverage_score": 0.78,
    "unique_views": 8,
    "top_view_captured": true,
    "blur_pass_ratio": 0.91,
    "exposure_pass_ratio": 0.88,
    "mask_confidence_avg": 0.76,
    "object_occupancy_avg": 0.23
  },

  "reconstruction": {
    "engine": "colmap_openmvs",
    "registered_images": 39,
    "sparse_points": 18420,
    "dense_points_fused": 410000,
    "mesher_used": "openmvs",
    "texturing_status": "real|degraded|absent",
    "attempt_score": 91.2
  },

  "export": {
    "has_uv": true,
    "has_material": true,
    "has_embedded_texture": true,
    "texture_integrity_status": "complete",
    "material_semantic_status": "diffuse_textured",
    "component_count": 2,
    "face_count": 48000
  },

  "labels": {
    "customer_ready": true,
    "draft_asset": false,
    "geometry_only": false,
    "failure_reason": null,
    "human_quality_rating": 4,
    "human_review_status": "approved"
  },

  "paths": {
    "original_frames_dir": "data/captures/cap_xxx/original_frames",
    "masked_frames_dir": "data/captures/cap_xxx/masked_frames",
    "masks_dir": "data/captures/cap_xxx/masks",
    "reports_dir": "data/captures/cap_xxx/reports",
    "final_glb": "data/registry/blobs/asset_xxx.glb"
  }
}
```

---

## 3. Device Metadata

```json
{
  "platform": "ios|android|web|unknown",
  "model": "unknown",
  "has_lidar": false,
  "camera_resolution": "1920x1080",
  "fps": 30
}
```

---

## 4. Capture Metrics

Required or best-effort fields:

- capture mode
- duration seconds
- original frame count
- accepted frame count
- capture score
- coverage score
- unique views
- top view captured
- blur pass ratio
- exposure pass ratio
- mask confidence average
- object occupancy average

---

## 5. Reconstruction Metrics

Recommended fields:

- engine
- registered images
- sparse points
- dense points fused
- mesher used
- texturing status
- attempt score
- selected attempt type
- fallback used

---

## 6. Export Metrics

Recommended fields:

- has UV
- has material
- has embedded texture
- texture integrity status
- material semantic status
- component count
- face count
- vertex count
- bbox
- ground offset

---

## 7. Label Taxonomy

### Asset State Labels

- `customer_ready`
- `draft_asset`
- `geometry_only`
- `texture_missing`
- `uv_missing`
- `mask_failed`
- `weak_sparse`
- `weak_dense`
- `table_contamination`
- `product_geometry_removed`
- `recapture_required`
- `ai_generated_fallback`
- `human_approved`
- `human_rejected`

### Failure Reason Enum

- `capture_blurry`
- `capture_underexposed`
- `capture_overexposed`
- `insufficient_views`
- `missing_top_view`
- `low_mask_confidence`
- `mask_contamination`
- `table_contamination`
- `weak_sparse_reconstruction`
- `weak_dense_reconstruction`
- `texturing_failed`
- `uv_missing`
- `embedded_texture_missing`
- `cleanup_removed_product`
- `glb_export_failed`
- `validation_failed`
- `ai_fallback_used`

---

## 8. Future Training Targets

The dataset can support the following targets:

### Binary Success Prediction

```text
Will this capture produce a customer-ready GLB?
```

### Multi-class Failure Classification

```text
Why did the capture/reconstruction fail?
```

### Next-best-action Recommendation

```text
What should the user do next?
```

### Segmentation Fine-tuning

```text
How can masks be improved for this product domain?
```

### Engine Routing

```text
Should this session use COLMAP/OpenMVS, recapture, AI draft, or human QA?
```