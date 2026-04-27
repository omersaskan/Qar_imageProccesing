# Phase 6.0: Evaluation Protocol

This protocol defines the standardized process for evaluating 3D reconstruction quality gains as we transition from deterministic to AI-enhanced modules.

## 1. Evaluation Dataset Structure
All evaluation data is stored in `datasets/evaluation/`:
- `videos/`: Raw input video files (.mp4, .mov).
- `ground_truth_masks/`: Manually labeled binary masks (.png) for selected validation frames.
- `metadata/`: JSON files containing object class, material properties, and known "gap" types.

## 2. Metadata Schema
Each evaluation video must have a corresponding metadata file in `datasets/evaluation/metadata/{video_name}.json`:
```json
{
  "video_id": "string",
  "object_class": "string",
  "gap_types": ["reflective", "low_feature", "occluded_bottom", "thin_structure"],
  "environment": "studio|office|outdoor",
  "lighting": "uniform|directional|low_light",
  "validation_frames": [
    {
      "frame_index": 120,
      "timestamp": "00:04.0",
      "gt_mask_path": "ground_truth_masks/{video_id}_f120.png"
    }
  ]
}
```

## 3. Metrics Definition

### A. Segmentation Metrics
*   **Mask IoU (Intersection over Union)**: measures spatial overlap between predicted and ground-truth masks.
    *   *Target*: IoU >= 0.92 for production ready.
*   **Background Leakage Ratio**: Percentage of predicted mask pixels that fall outside the ground truth area.
    *   *Target*: <= 2%.

### B. Reconstruction Integrity Metrics
*   **Dense Mask Exact Match**: Ratio of dense images with an exact filename match in the mask directory.
*   **Dense Mask Dimension Match**: Ratio of dense masks with dimensions matching their corresponding images.
*   **Fallback White Ratio**: Ratio of masks that defaulted to full-white due to poor quality or missing data.

### C. Pipeline Performance
*   **Reconstruction Success Rate**: Percentage of jobs resulting in `production_ready` or `review_ready`.
*   **Isolation Confidence**: The total score of the primary component (0.0 to 1.0).
*   **Hole Area Reduction**: (For Phase 6.2+) Comparative reduction in un-reconstructed mesh surface area.

## 4. Before/After Comparison Protocol
When a new AI module (e.g., SAM2) is introduced:
1.  **Baseline Run**: Run the latest deterministic version on the evaluation dataset.
2.  **Experimental Run**: Run the version with the AI module integrated.
3.  **Delta Analysis**:
    *   Calculate mean improvement in IoU.
    *   Measure impact on `production_ready` conversion rate.
    *   Quantify compute overhead (latency/GPU usage).
4.  **Visual Audit**: Side-by-side comparison of GLB outputs focusing on the targeted "gap."

## 5. Acceptance for Phase 6.1
Phase 6.0 is considered complete when:
- [ ] 5+ real videos are curated in the dataset.
- [ ] Ground truth masks are labeled for at least 3 frames per video.
- [ ] Baseline metrics are calculated for the current main branch.
- [ ] The `scripts/evaluate_segmentation.py` and `scripts/evaluate_baselines.py` are verified to work.

---

## 🛑 User Review Required
**Review of this evaluation protocol is required before proceeding to Phase 6.1 (SAM2 Integration).**
