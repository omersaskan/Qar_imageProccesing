# Capture Quality Scoring

## 1. Purpose

The customer-facing capture workflow must not rely on a fixed video duration.

A capture should be accepted only when the system has enough evidence that the input can produce a reliable 3D asset.

The main objective is:

```text
Finish capture when quality is sufficient.
Do not finish capture only because a timer ended.
```

This enables short captures for easy objects and longer guided captures for difficult objects.

---

## 2. Live Metrics

The guided capture client should collect or estimate the following metrics.

### 2.1 Blur Score

Measures frame sharpness.

Common proxy:

- Laplacian variance
- edge density
- high-frequency energy

Low blur score means the user is moving too fast or the camera is out of focus.

### 2.2 Exposure Score

Detects underexposed or overexposed frames.

Signals:

- mean luminance
- histogram clipping
- percentage of very dark / very bright pixels

### 2.3 Object Occupancy

Measures how much of the frame is occupied by the object mask.

Too small:

- low reconstruction detail
- weak texture resolution

Too large:

- object may be clipped
- insufficient background context
- unstable framing

### 2.4 Object Clipped

Detects whether the mask touches frame borders.

A clipped object should not count as a high-quality viewpoint.

### 2.5 Mask Confidence

Measures segmentation reliability.

Can be derived from:

- segmentation backend confidence
- mask shape consistency
- dominant component ratio
- solidity
- occupancy stability

### 2.6 Mask Purity

Measures whether the mask mostly contains the product rather than table, hand, background, or support geometry.

### 2.7 Support / Table Suspicion

Detects whether large horizontal or lower-frame regions are included in the object mask.

Important for:

- plates
- food
- cups
- products placed on tables

### 2.8 Motion Speed

Detects fast movement.

Possible proxies:

- optical flow magnitude
- frame-to-frame object center movement
- motion blur
- IMU data where available

### 2.9 Feature Richness Proxy

Estimates whether COLMAP is likely to find enough stable features.

Possible proxies:

- corner count
- ORB/SIFT-like feature count
- texture entropy
- edge density inside mask

### 2.10 Optical Flow / Parallax Proxy

Estimates whether the camera is actually moving around the object rather than only shaking in place.

### 2.11 Unique View Sector Count

Counts how many distinct viewpoint sectors were captured.

### 2.12 Horizontal Orbit Coverage

Measures whether the user captured enough views around the object horizontally.

### 2.13 Elevated / Top Orbit Coverage

Measures whether the user captured a top or elevated angle pass.

Important for:

- plates
- food
- bowls
- cups
- flat objects

### 2.14 Center Stability

Measures whether the object remains centered.

Unstable framing reduces reconstruction reliability.

### 2.15 Scale Variation

Measures whether the object changes apparent size across views.

Some variation is useful as a proxy for camera movement and elevation.

### 2.16 Aspect Variation

Measures whether the object silhouette changes across views.

Useful as a weak proxy for viewpoint diversity.

---

## 3. Capture Readiness Score

Suggested score range:

```text
0 to 100
```

### Score Bands

| Score | Status | Action |
|---:|---|---|
| 0-39 | Reject | Require recapture |
| 40-59 | Weak | Continue capture or recapture |
| 60-74 | Acceptable draft | Allow internal draft, not ideal |
| 75-89 | Production candidate | Accept |
| 90-100 | Strong capture | Accept confidently |

### Suggested Formula

```text
capture_score =
  0.15 * frame_quality_score
+ 0.15 * mask_quality_score
+ 0.20 * viewpoint_coverage_score
+ 0.15 * parallax_score
+ 0.10 * object_framing_score
+ 0.10 * feature_richness_score
+ 0.10 * elevated_view_score
+ 0.05 * stability_score
```

Each sub-score should be normalized to 0-100.

---

## 4. Capture Ring Model

The UI should show a ring or level system around the object.

Suggested sectors:

```text
front-left
front
front-right
right
back-right
back
back-left
left
elevated/top pass
```

A sector fills only when:

- frame is sharp enough
- exposure is acceptable
- object is not clipped
- mask confidence is sufficient
- object occupancy is within range
- viewpoint is sufficiently different from previous accepted views

The elevated/top pass is separate because many products require top-surface coverage.

---

## 5. Minimum Gates

Capture should not be accepted until all minimum gates pass.

Suggested gates:

| Gate | Requirement |
|---|---|
| readable frames | At least 8 |
| unique views | At least 5 |
| blur pass ratio | At least 0.75 |
| clipped frame ratio | At most 0.20 |
| average mask confidence | At least 0.60 |
| object occupancy | Within configured range |
| elevated view evidence | Required for flat/food/plate categories |
| support suspicion | Must not be severe |

---

## 6. Client-to-Server Contract

The guided capture client should submit:

```text
original_frames/
masked_frames/
masks/
thumbnails/
capture_report.json
device_metadata.json
```

### Required Metadata

```json
{
  "capture_mode": "guided_camera",
  "duration_sec": 42,
  "accepted_frame_count": 42,
  "raw_frame_count": 120,
  "capture_score": 84.5,
  "coverage_score": 0.81,
  "unique_view_count": 8,
  "top_view_captured": true,
  "blur_pass_ratio": 0.91,
  "exposure_pass_ratio": 0.87,
  "mask_confidence_avg": 0.78,
  "object_occupancy_avg": 0.24,
  "product_category": "plate_food"
}
```

### Optional Metadata

```json
{
  "platform": "ios",
  "device_model": "unknown",
  "has_lidar": false,
  "ar_pose_available": true,
  "depth_available": false,
  "camera_resolution": "1920x1080",
  "fps": 30
}
```

---

## 7. Operator Guidance

If capture fails, the system should provide actionable feedback:

| Failure | User Guidance |
|---|---|
| blurry frames | Move slower |
| underexposed | Improve lighting |
| object clipped | Move farther back |
| low occupancy | Move closer |
| low unique views | Continue moving around the object |
| missing top view | Capture from a higher angle |
| low mask confidence | Use a cleaner background |
| table contamination | Use a contrasting surface |