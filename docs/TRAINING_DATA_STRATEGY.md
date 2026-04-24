# Training Data Strategy

## 1. Why Every Capture Must Be Dataset-Ready

Every capture session is not only a reconstruction job. It is also a future training-data candidate.

The system should not wait until the fine-tuning phase to start collecting structured data. Instead, every capture should produce a privacy-safe training manifest that records capture quality, reconstruction outcome, export quality, validation result, labels, and training eligibility.

The immediate goal is not to train a model today.

The immediate goal is:

```text
Collect high-quality structured data now,
so future ML and fine-tuning can be done reliably.
```

---

## 2. Target ML Use Cases

Future ML systems may include:

### Capture Success Predictor

Predict whether a capture is likely to produce a customer-ready GLB.

### Failure Reason Classifier

Classify why a session failed:

- blur
- poor lighting
- insufficient views
- weak sparse reconstruction
- texture failure
- UV missing
- table contamination

### Guidance Recommender

Suggest the next best user action:

- move slower
- add top angle
- improve lighting
- move closer
- use cleaner background
- recapture

### Segmentation Fine-tuning

Use original frames and masks to improve object segmentation for product, food, plate, cup, and packaging categories.

### Product Category Preset Optimizer

Learn which capture and cleanup settings work best for each product type.

### Reconstruction Parameter Optimizer

Predict which reconstruction settings should be used for a given capture.

### AI Fallback Router

Predict whether the system should use:

- own photogrammetry
- recapture
- AI draft fallback
- human QA

---

## 3. What to Store Per Session

Each session should store or reference:

```text
original frames
masked frames
masks
thumbnails
raw video pointer
capture_report.json
quality_report.json
coverage_report.json
reconstruction_audit.json
cleanup_stats.json
export_metrics.json
validation_report.json
user_feedback.json
human_review.json
final_asset_metadata.json
training_manifest.json
```

---

## 4. What Not to Store Forever

The following should not be retained indefinitely by default:

- raw video
- personal identifiers
- device identifiers that can identify a person
- EXIF location data
- user account data inside training manifests
- billing or business identity data

Raw videos should have short configurable retention.

Training manifests should contain privacy-safe derived data.

---

## 5. Labels

The system should generate structured labels.

### Asset Labels

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

### Failure Reason Labels

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

## 6. Dataset Versioning

Future dataset exports must be versioned.

Each dataset export should include:

- dataset version
- created at timestamp
- source session IDs
- train / validation / test split
- label distribution
- product category distribution
- device distribution
- capture mode distribution
- consent summary
- exclusion rules

Example:

```json
{
  "dataset_version": "v2026-04-001",
  "created_at": "2026-04-24T12:00:00Z",
  "source_session_count": 1200,
  "train_count": 840,
  "val_count": 180,
  "test_count": 180,
  "eligible_only": true
}
```

---

## 7. Consent and Privacy

Training eligibility is not automatic.

Consent states:

| Consent Status | Meaning | Eligible |
|---|---|---|
| `unknown` | No explicit training consent | No |
| `denied` | User denied training use | No |
| `granted` | User granted training use | Yes |
| `internal_only` | Internal test data only | Depends on policy |

Rules:

- Raw user identity must not be written to training manifests.
- `product_id` should be hashed.
- EXIF/location metadata should be stripped or ignored.
- Deletion requests must revoke training eligibility.
- Training data must be auditable.

---

## 8. Minimum Acceptance Criteria

The training data system is acceptable when:

- Every completed, failed, and recapture session can produce a training manifest.
- The manifest includes capture, reconstruction, export, and validation sections.
- The manifest includes labels.
- The manifest includes consent and eligibility.
- The manifest does not contain raw user identity.
- A global training registry index is updated.
- Missing reports do not crash manifest generation.