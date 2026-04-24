# AI Fallback Router

## 1. Purpose

TRELLIS, Meshy, or other image-to-3D systems are not the primary reconstruction engine for faithful product scans.

They are fallback or draft-generation systems.

The primary trusted engine remains:

```text
Guided capture + COLMAP + OpenMVS
```

AI fallback should be used when a generated estimate is acceptable or when the system needs to provide a quick internal preview.

---

## 2. When to Use AI Fallback

AI fallback may be used when:

- the object has low visual texture
- COLMAP registration is weak
- OpenMVS texturing fails
- user requests a quick preview
- internal demo output is needed
- exact geometry is not required
- non-critical asset workflow is selected
- photogrammetry output is draft-only

---

## 3. When Not to Use AI Fallback

AI fallback should not be used as final output when:

- exact product fidelity is required
- branded packaging or logos must be accurate
- dimensions must be reliable
- menu item geometry must match the real object
- the customer expects a faithful scan
- regulatory or commercial accuracy matters

---

## 4. Metadata Requirements

AI-generated assets must be explicitly labeled.

### AI Asset Metadata

```json
{
  "source_engine": "trellis_or_meshy",
  "fidelity_type": "generated_estimate",
  "not_exact_scan": true,
  "needs_human_review": true,
  "confidence_score": 0.72
}
```

### Photogrammetry Asset Metadata

```json
{
  "source_engine": "colmap_openmvs",
  "fidelity_type": "reconstructed_scan",
  "not_exact_scan": false,
  "texture_status": "complete",
  "uv_status": "present",
  "capture_score": 84.2
}
```

---

## 5. Router Decision Matrix

| Capture / Reconstruction State | Recommended Action |
|---|---|
| Strong capture | Own COLMAP/OpenMVS pipeline |
| Medium capture | Own pipeline + possible AI draft preview |
| Weak capture | Recapture |
| Urgent preview | AI draft fallback |
| Texture failure | Retry texturing, then AI draft if allowed |
| Exact fidelity required | No AI final output |
| Internal demo | AI draft allowed |

---

## 6. Required Labels

AI fallback outputs must be labeled with:

- `ai_generated_fallback`
- `generated_estimate`
- `not_exact_scan`
- `needs_human_review`

They must not be silently mixed with faithful scan assets.

---

## 7. Future Work

P2 may include:

- TRELLIS integration
- Meshy integration
- fallback cost tracking
- fallback quality scoring
- human review workflow
- side-by-side comparison between AI draft and photogrammetry result