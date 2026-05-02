# Phase 3B — Depth Studio Report

> Implementation-only mode. No test suite changes. 638/638 existing tests still passing.

---

## Yeni Modüller (`modules/depth_studio/`)

| Dosya | Sorumluluk |
|-------|-----------|
| `__init__.py` | Package marker |
| `input_router.py` | File extension → `image` / `video` routing |
| `image_preflight.py` | cv2 / PIL ile format, boyut, bütünlük kontrolü |
| `video_frame_selector.py` | Laplacian blur + merkez içerik skoru ile en iyi frame seçimi |
| `depth_provider_base.py` | Abstract `DepthProviderBase` — `is_available()` + `safe_infer()` |
| `depth_anything_provider.py` | Depth Anything V2 (native dpt + transformers fallback) |
| `depth_pro_provider.py` | Apple Depth Pro (experimental, `DEPTH_PRO_ENABLED=false` guard) |
| `depth_output.py` | 16-bit PNG yaz/oku, EXR desteği, renk haritası preview |
| `depth_refinement.py` | Median edge cleanup, bilateral, guided filter (approximate) |
| `depth_to_mesh.py` | Subdivided relief plane + UV — trimesh GLB export |
| `texture_projection.py` | Kaynak görüntüyü texture olarak hazırla (resize, JPEG) |
| `glb_builder.py` | depth map + texture → GLB pipeline entry point |
| `manifest.py` | `depth_studio_manifest.json` builder + writer |
| `pipeline.py` | Top-level orchestrator — 8 adım zinciri |

---

## Yeni API Endpointleri (`modules/operations/api.py`)

| Method | Path | Açıklama |
|--------|------|---------|
| `POST` | `/api/depth-studio/upload` | Image veya video yükle, session oluştur |
| `POST` | `/api/depth-studio/process/{session_id}` | Tam pipeline'ı çalıştır |
| `GET` | `/api/depth-studio/status/{session_id}` | Anlık durum özeti |
| `GET` | `/api/depth-studio/manifest/{session_id}` | Tam manifest JSON |
| `GET` | `/api/depth-studio/preview/{session_id}` | Depth preview PNG (FileResponse) |

Tüm endpointler `verify_api_key` ile korumalı. `DEPTH_STUDIO_ENABLED=false` iken `/upload` 503 döner.

---

## Yeni UI Dosyaları

| Dosya | İçerik |
|-------|--------|
| `ui/depth_studio.html` | Tam Depth Studio arayüzü |
| `ui/index.html` (güncellendi) | Nav'a "⬡ Depth Studio" linki eklendi |

Depth Studio UI özellikleri:
- Provider seçimi (Depth Anything V2 / Depth Pro)
- Sürükle-bırak veya dosya seçici (image + video)
- Upload → process → sonuç tek akış
- Depth preview imajı (API'den çekilir)
- Output grid: provider, model, format, face count, GLB path
- Warning tag'leri (single_view_geometry, preview_only_asset vb.)
- Depth Pro seçilince lisans uyarısı gösterimi

---

## Feature Flags (varsayılan hepsi `false`)

```
SINGLE_IMAGE_DEPTH_ENABLED=false
DEPTH_STUDIO_ENABLED=false
DEPTH_STUDIO_ALLOW_VIDEO_INPUT=true
DEPTH_STUDIO_DEFAULT_PROVIDER=depth_anything_v2
DEPTH_PRO_ENABLED=false
DEPTH_OUTPUT_FORMAT=png16
DEPTH_OUTPUT_ALLOW_EXR=true
DEPTH_MESH_MODE=relief_plane
DEPTH_GRID_RESOLUTION=256
DEPTH_EDGE_CLEANUP_ENABLED=true
DEPTH_PREVIEW_ONLY=true
DEPTH_STUDIO_REQUIRE_EXPLICIT_FINAL_OVERRIDE=true
```

---

## Manifest Yapısı (`depth_studio_manifest.json`)

```json
{
  "enabled": true,
  "mode": "depth_studio",
  "session_id": "ds_abc123",
  "created_at": "2026-05-02T...",
  "provider": "depth_anything_v2",
  "model_name": "depth-anything/Depth-Anything-V2-Small-hf",
  "provider_status": "ok",
  "license_note": "Depth Anything V2 — Apache 2.0",
  "input_type": "image",
  "input_path": "...",
  "selected_frame_path": null,
  "depth_map_path": "derived/depth_16.png",
  "depth_format": "png16",
  "refinement_applied": true,
  "mesh_mode": "relief_plane",
  "mesh_vertex_count": 65536,
  "mesh_face_count": 130050,
  "glb_path": "derived/preview_mesh.glb",
  "status": "ok",
  "warnings": ["single_view_geometry", "backside_not_observed", "preview_only_asset"],
  "is_true_3d": false,
  "has_backside": false,
  "preview_only": true,
  "explicit_final_override_required": true
}
```

---

## Session Klasör Yapısı

```
data/depth_studio/{session_id}/
  input/
    upload.jpg          ← orijinal input
  derived/
    selected_frame.jpg  ← video ise seçilen frame
    depth_16.png        ← raw 16-bit depth
    depth_refined_16.png← edge cleanup sonrası (opsiyonel)
    depth_preview.png   ← 8-bit colormapped preview
    texture.jpg         ← resize edilmiş texture
    preview_mesh.glb    ← final output
  manifests/
    depth_studio_manifest.json
```

---

## Bilinçli Sınırlar

- **Provider gerçek model yüklemesi**: `depth_anything_v2` ve `depth_pro` paketleri kurulu değilse `status=unavailable` döner, crash olmaz.
- **Video input**: Best-frame seçimi blur + merkez içerik skoru ile yapılır; temporal coherence veya subject detection yok.
- **Mesh**: Relief plane (subdivided grid). Backside geometry yok, `has_backside=false`.
- **Texture baking**: Basit UV projection — gelişmiş material baking yok.
- **EXR**: `pyopenexr` kurulu değilse EXR yazımı RuntimeError fırlatır (graceful olarak yakalanmaz, optional feature).
- **Production publish**: `explicit_final_override_required=true`, bu asset AR gate'ten production olarak geçmez.
- **Test coverage**: Bu fazda test yazılmadı (Phase 3B kuralı).

---

## Sonraki Faz Önerisi (Phase 3C)

1. **Model inference smoke tests** — `safe_infer()` mock'larla, gerçek model indirmeden
2. **RC integration tests** — `test_depth_studio_pipeline.py`, `test_depth_studio_api.py`
3. **Gerçek model download** — Depth Anything V2 Small checkpoint (`~100MB`)
4. **Video best-frame kalitesi** — Subject detection (segmentation mask ile)
5. **Multi-scale depth fusion** — Birden fazla input imajı varsa ortalama depth
6. **AR gate entegrasyonu** — `preview_only` asset için ayrı `depth_studio_gate` verdict
