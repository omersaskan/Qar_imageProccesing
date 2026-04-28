# Visual Audit Report: frame_0010
**Date:** 2026-04-28
**Subject:** Investigation of "Hard Failure" in SAM2/Legacy Segmentation

## 1. Visual Evidence
Overlays for `frame_0010` (Capture `cap_29ab6fa1`):

| Type | Visual State | Observation |
| :--- | :--- | :--- |
| **GT Mask** | ![GT Overlay](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/Qar_imageProccesing/results/frame_0010_audit/frame_0010_gt_overlay.jpg) | **CRITICAL ERROR**: GT mask is on a green pillow in the background. |
| **Legacy (Rembg)** | ![Legacy Overlay](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/Qar_imageProccesing/results/frame_0010_audit/frame_0010_legacy_overlay.jpg) | Correctly identified the bottle, but IoU=0.0 due to GT mismatch. |
| **SAM2 Large** | ![SAM2 Large Overlay](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/Qar_imageProccesing/results/frame_0010_audit/frame_0010_sam2_large_overlay.jpg) | Correctly identified the bottle with high precision. |

## 2. Audit Questionnaire
- **Is the GT mask aligned with frame_0010?**
  **NO**. The GT mask for `frame_0010` incorrectly labels a background object (green pillow/chair) instead of the subject (water bottle).
- **Is the object visible, clipped, blurred, reflective, or occluded?**
  The object (water bottle) is clearly visible and centered. It is slightly reflective (plastic), which SAM2 handles well.
- **Did legacy select the wrong region or no useful region?**
  Legacy selected the **correct** region (the bottle). Its IoU was 0 only because of the faulty GT.
- **Did SAM2 select the wrong region, partial object, or background?**
  SAM2 selected the **correct** region (the bottle) with better edge adherence than legacy.
- **Is this a data/GT problem or a segmentation/prompt problem?**
  This is a **DATA/GT problem**. The evaluation metrics for this frame are invalid.

## 3. Impact on Evaluation
The "failure" of SAM2 to reach the +0.05 IoU gain target is partly due to this GT error. In frames where the GT is correct (e.g., `frame_0000`), SAM2 already matches or exceeds legacy performance. However, because `frame_0010` is 1 of only 3 GT frames, its "0 IoU" significantly drags down the mean.

**Recommendation:** Fix the GT for `frame_0010` before final rejection of image-mode. However, video-temporal SAM2 remains the superior architectural choice for robustness.
