# Phase F — UI + WebGL Review Experience

Phase F extends the dashboard from a basic AI preview UI into a WebGL review workbench.

The goal is to review photogrammetry output, AI preview output, capture coverage diagnostics, and AI-vs-photogrammetry differences without mixing any of this with production publishing.

## Hard rule

Phase F is review-only and non-blocking.

It must not:

- modify `GateValidator`
- modify `MediaRecorder`
- modify `finishRecording`
- alter quality manifest creation or validation
- alter capture HUD accept/reject logic
- alter production asset card publish state
- change `session.status`
- change `publish_state`
- call `registry.publish_asset()`
- call `registry.set_active_version()`
- overwrite photogrammetry manifests
- clear `recapture_required`
- change registry active pointer

## Phase F.0 — Config and review data endpoints

Before building WebGL UI, expose read-only data endpoints.

### `GET /api/config`

Returns dashboard feature flags.

Example:

```json
{
  "ai_3d_preview_enabled": false,
  "ai_3d_provider": "none",
  "sam3d_enabled": false,
  "meshy_enabled": false
}
```

### `GET /api/sessions/{session_id}/review-assets`

Returns asset metadata needed by the viewer.

Example:

```json
{
  "session_id": "cap_123",
  "photogrammetry": {
    "exists": true,
    "mesh_path": ".../final.glb",
    "geometry_source": "photogrammetry",
    "production_status": "production_candidate",
    "file_size_bytes": 1234567,
    "polycount": 102345
  },
  "ai_preview": {
    "exists": true,
    "mesh_path": ".../reports/ai_proxy_preview.glb",
    "geometry_source": "meshy",
    "production_status": "review_ready",
    "requires_manual_review": true,
    "warning_labels": [
      "AI-generated preview. Not production coverage.",
      "Does not bypass recapture.",
      "Manual review required.",
      "Review-only artifact."
    ],
    "file_size_bytes": 934455,
    "polycount": 80432
  }
}
```

This endpoint is read-only and must not change session or registry state.

### `GET /api/sessions/{session_id}/coverage-debug`

Returns capture coverage debug data for the post-session coverage viewer.

Example:

```json
{
  "session_id": "cap_123",
  "coverage_percent": 91.2,
  "max_gap_deg": 38.0,
  "missing_angles": [120, 130, 140],
  "accepted_frames": [
    {
      "frame": "frame_001.jpg",
      "azimuth": 12.4,
      "confidence": 0.91,
      "accepted": true
    },
    {
      "frame": "frame_002.jpg",
      "azimuth": 25.0,
      "confidence": 0.42,
      "accepted": false,
      "reasons": ["blur"]
    }
  ],
  "heatmap": [
    {"sector": 0, "score": 0.9},
    {"sector": 1, "score": 0.8}
  ]
}
```

### `GET /api/sessions/{session_id}/comparison-data`

Returns data for AI-vs-photogrammetry comparison.

May include:

- photogrammetry mesh info
- AI preview mesh info
- warning labels
- manual review notes if available
- evaluation summary if available

## Optional debug artifact

For Phase F.2, add this optional report:

```text
captures/{session_id}/reports/capture_debug_manifest.json
```

Recommended fields:

- per-frame azimuth
- accepted/rejected flag
- confidence
- rejection reasons
- blur score if available
- sector mapping

This must be a debug artifact only. It must not become a production gate.

## Phase F.1 — WebGL Review Viewer

### Goal

Render session 3D assets in the browser for review.

### Required features

- GLB viewer
- photogrammetry output card
- AI preview output card
- bbox overlay
- pivot overlay
- ground plane overlay
- texture toggle
- wireframe toggle
- orbit / zoom / pan
- reset camera
- model stats panel

### Cards

Cards must be visually distinct.

#### Photogrammetry Production Candidate

Show:

- status badge
- geometry source = `photogrammetry`
- production candidate badge
- polycount
- file size
- viewer canvas
- toggles for bbox, pivot, ground plane, wireframe, texture

#### AI Preview / Review Only

Show:

- provider badge: `meshy`, `sam3d`, or `none`
- review-only badge
- requires-manual-review badge
- warning area
- polycount
- file size
- viewer canvas
- same toggles

### Required AI warning labels

AI preview UI must visibly show:

```text
AI-generated preview. Not production coverage.
Does not bypass recapture.
Manual review required.
Review-only artifact.
```

## Phase F.2 — Capture Coverage Debug Viewer

### Goal

Make capture quality and recapture reasons visually understandable.

### Required features

- accepted frame camera ring
- missing angles
- coverage heatmap
- frame confidence markers
- rejection reason visualization

### Sections

#### Coverage Ring

- 360-degree ring
- accepted frames as positive markers
- rejected frames as warning markers
- missing arcs as highlighted sectors

#### Coverage Heatmap

- sector score visualization
- blur density
- low-confidence clusters

#### Summary

Show:

- coverage percentage
- max gap
- accepted frame count
- total frame count if available
- rejection breakdown

#### Missing Angles

Example:

```text
110°-145°
280°-300°
```

## Phase F.3 — AI vs Photogrammetry Comparison

### Goal

Help reviewers compare AI preview output against photogrammetry output.

### Required features

- side-by-side viewer
- synchronized cameras
- silhouette comparison
- warning labels
- manual review notes

### Layout

Left:

- photogrammetry viewer

Right:

- AI preview viewer

Side or bottom panel:

- warning labels
- geometry source differences
- polycount / file size differences
- generation time / cost if available
- review notes
- evaluation summary if available

### Silhouette comparison

Initial implementation can be simple:

- render both models from the same camera angle
- render flat silhouette pass
- overlay or diff silhouettes on canvas

This is for reviewer assistance only and must not produce a production decision.

## Manual review notes

Initial Phase F.3 may render notes read-only from:

- `reports/ai_review_audit.jsonl`
- a future review notes file
- evaluation summary markdown

Editing notes should stay closer to Phase E manual review endpoints unless explicitly implemented there.

## Recommended UI file structure

Do not put all WebGL logic directly into `ui/app.js`.

Recommended structure:

```text
ui/
  index.html
  app.js
  styles.css
  webgl/
    viewer_core.js
    review_viewer.js
    coverage_debug_viewer.js
    comparison_viewer.js
    overlays.js
    loaders.js
    ui_panels.js
```

If no bundler is present, use plain ES modules and import them from `index.html`.

## Recommended library

Use `three.js` for WebGL review tooling.

Recommended helpers:

- `GLTFLoader`
- `OrbitControls`
- `BoxHelper`
- `AxesHelper`
- `GridHelper`

`model-viewer` is simpler but is not ideal for bbox overlays, synchronized viewers, wireframe toggles, and silhouette comparison.

## Phase F tests

### Phase F.1 tests

```python
def test_review_assets_endpoint_returns_distinct_cards(): ...
def test_photogrammetry_card_is_separate_from_ai_preview_card(): ...
def test_ai_preview_warning_labels_visible(): ...
def test_webgl_viewer_handles_missing_glb_gracefully(): ...
def test_texture_wireframe_toggle_state_changes(): ...
def test_overlay_toggle_bbox_pivot_ground_plane(): ...
```

### Phase F.2 tests

```python
def test_coverage_debug_endpoint_returns_heatmap(): ...
def test_missing_angles_rendered_when_present(): ...
def test_frame_confidence_markers_rendered(): ...
def test_coverage_debug_handles_missing_manifest_gracefully(): ...
```

### Phase F.3 tests

```python
def test_side_by_side_viewer_renders_both_assets(): ...
def test_silhouette_comparison_panel_visible(): ...
def test_warning_labels_visible_in_comparison_view(): ...
def test_manual_review_notes_render_if_present(): ...
def test_comparison_view_non_blocking_if_ai_preview_missing(): ...
```

If jsdom is insufficient, use Playwright smoke tests for:

- viewer container render
- distinct photogrammetry and AI cards
- warning labels
- toggles
- non-blocking missing-AI behavior

## Completion criteria

Phase F is complete only when:

- config endpoint exists
- review asset endpoint exists
- coverage debug endpoint exists
- comparison data endpoint exists
- photogrammetry and AI cards are visually distinct
- WebGL GLB viewer works for available assets
- AI preview warnings are visible
- missing AI preview is non-blocking
- coverage debug viewer renders available debug data
- comparison viewer renders both assets when present
- existing upload/capture regression tests still pass
