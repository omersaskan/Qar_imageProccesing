# Sprint 1 — Measurement Foundation + Camera Model Resolver + Reconstruction Scorecard

> **Status:** ✅ Complete · **Tests:** 33/33 passed · **Regression:** 0 new

## Hedef
"İyi mi kötü mü?" sorusuna sayısal cevap. Sprint 2-8 için ön koşul:
sayısal metrik altyapısı olmadan kalite iyileştirmeleri optimizasyonsuz olur.

Üç paralel çıkartı:
1. **Camera model resolver** — RADIAL hardcoded yerine EXIF/HFOV/device-DB ile dinamik seçim
2. **Coverage + geometric metrics** — observed surface, azimuth/elevation, manifold/hole/component
3. **Reconstruction scorecard** — her job için tek bir `quality_report.json` (12-15 metrik + grade)

## Eklenen / Değişen Dosyalar

### Yeni modüller
| Dosya | LoC | İçerik |
|-------|-----|--------|
| [`modules/reconstruction_engine/camera_model_resolver.py`](../modules/reconstruction_engine/camera_model_resolver.py) | 248 | EXIF reader + 9-row device DB + HFOV bucket selector + filename hint fallback |
| [`modules/qa_validation/coverage_metrics.py`](../modules/qa_validation/coverage_metrics.py) | 175 | 8-bucket azimuth, 5-bucket elevation, 3-bucket multi-height, view diversity |
| [`modules/qa_validation/geometric_quality.py`](../modules/qa_validation/geometric_quality.py) | 218 | watertight, manifold, hole area, edge length, aspect ratio, component share, A/B/C/F grade |
| [`modules/qa_validation/scorecard.py`](../modules/qa_validation/scorecard.py) | 180 | Schema v1 — coverage + geometry + texture + reconstruction + capture/color profile + overall grade + blockers |

### Değişen dosyalar
| Dosya | Değişiklik |
|-------|-----------|
| [`adapter.py:128-167`](../modules/reconstruction_engine/adapter.py#L128) | `feature_extractor()` artık `camera_model` parametresi alıyor (default `RADIAL`) |
| [`adapter.py:1220-1240`](../modules/reconstruction_engine/adapter.py#L1220) | COLMAP_DENSE path: resolver çağrılıp seçilen model geçirilir, log'a yazılır |
| [`adapter.py:1600-1625`](../modules/reconstruction_engine/adapter.py#L1600) | OpenMVS path: aynı resolver entegrasyonu |
| [`runner.py:_finalize_best_attempt`](../modules/reconstruction_engine/runner.py) | Job sonu `build_scorecard()` + `write_scorecard()` çağrılıyor; manifest yanına `quality_report.json` |
| [`scripts/smoke_check.sh`](../scripts/smoke_check.sh) | Stage 2'ye Sprint 1 modüllerinin import doğrulaması eklendi |

### Yeni testler
| Dosya | Test sayısı | Kapsam |
|-------|------------|--------|
| `tests/sprint1/test_camera_model_resolver.py` | 11 | HFOV math, EXIF parsing (IFDRational/tuple/35mm field), iPhone main/ultrawide DB, HFOV fallback, filename hint, mixed-lens detection |
| `tests/sprint1/test_coverage_metrics.py` | 8 | Full ring, half ring, empty, multi-height (1/2/3 band), heuristic mesh observed |
| `tests/sprint1/test_geometric_quality.py` | 6 | Clean box=A, sphere compactness 0.45-0.62, open mesh holes, 3 components→demote, schema completeness |
| `tests/sprint1/test_scorecard.py` | 6 | Skeleton, write+read, manifest pickup, blockers, corrupt manifest resilience, clean-box grading |

**Toplam: 31 yeni test + 2 reuse → 33/33 passed.**

## Mimari Etki

### Camera Model Resolver — Beklenen Kalite Etkisi

Önceki kod tüm cihazlar için `RADIAL` (2 distortion params, ~70° HFOV optimum):

| Cihaz / lens | HFOV | RADIAL fit | Yeni resolver | Tahmini kazanç |
|--------------|------|-----------|----------------|---------------|
| iPhone main (26mm) | ~70° | OK | RADIAL (DB hit) | nötr |
| iPhone ultrawide (13mm) | ~120° | **kötü** — kenar feature drift | OPENCV_FISHEYE | **+30-50% registered images** |
| Pixel ultrawide (14mm) | ~107° | sınırda | OPENCV | +15-30% |
| GoPro / action cam | ~120° | kötü | OPENCV_FISHEYE | +40-60% |
| DSLR / mirrorless | ~50° | OK | RADIAL | nötr |
| Mixed-lens capture | mix | parçalanır | OPENCV (fallback) | +15-25% |

EXIF okuma fail-safe: hiçbir sinyal yoksa `RADIAL` (mevcut davranış korunur).

### Scorecard — Validator Bağımsız Layer

`quality_report.json` mevcut `ValidationReport`'u **değiştirmez**, **paralel** çalışır. Sprint 2'de validator buradan beslenecek. Şu an:

- Her başarılı reconstruction `manifest.json` yanına `quality_report.json` yazıyor
- Schema v1 sabit; gelecek sprintlerde `schema_version` bump ile genişler
- Eksik input → graded F + blocker listesi (silent fail yok)
- Manifest bozuk olsa bile crash yok (warning log + boş profil)

### Yeni `quality_report.json` Şeması (örnek)

```json
{
  "schema_version": 1,
  "job_id": "job_cap_xxx",
  "generated_at": "2026-05-02T12:34:56+00:00",
  "coverage": {
    "sample_count": 26,
    "observed_surface_ratio": 0.78,
    "observed_surface_method": "point_cloud",
    "azimuth_coverage_ratio": 0.875,
    "azimuth_buckets_filled": 7,
    "max_azimuth_gap_deg": 52.0,
    "elevation_coverage_ratio": 0.4,
    "multi_height_score": 0.333,
    "multi_height_buckets": {"low": 26, "mid": 0, "top": 0},
    "view_diversity_score": 0.52
  },
  "geometry": {
    "vertex_count": 56234,
    "face_count": 137364,
    "is_watertight": false,
    "manifold_ratio": 0.92,
    "hole_area_ratio": 0.04,
    "aspect_ratio_p99": 12.4,
    "volume_to_bbox_ratio": 0.28,
    "component_count": 1,
    "largest_component_face_share": 1.0,
    "grade": "B",
    "grade_reasons": ["aspect_ratio_p99 12 (sliver triangles)"]
  },
  "texture": { "status": "pass", "black_pixel_ratio": 0.08, ... },
  "reconstruction": {
    "engine_used": "colmap_dense",
    "elapsed_seconds": 412.3,
    "score": 19310.4
  },
  "capture_profile": { "preset_key": "small_on_surface", ... },
  "color_profile": { "category": "white_cream", ... },
  "overall": {
    "grade": "B",
    "production_ready": false,
    "review_required": true,
    "blockers": ["multi_height_score <0.34 — only one elevation band captured"]
  }
}
```

## Başarı Kriterleri

| # | Kriter | Sonuç |
|---|--------|-------|
| 1 | EXIF okur 6+ cihaz tipini doğru kategorize eder | ✅ 9-row DB + HFOV fallback (test'te 6 cihaz) |
| 2 | Wide-angle (>75° HFOV) için OPENCV/OPENCV_FISHEYE seçilir | ✅ HFOV bucket testleri |
| 3 | Boş input / corrupt EXIF → RADIAL fallback (panik yok) | ✅ 4 negative test |
| 4 | Coverage rapor: azimuth + elevation + multi-height en az 12 metrik | ✅ 12 metrik |
| 5 | Geometric rapor: watertight + manifold + holes + edges + components | ✅ 21 alan, A/B/C/F grade |
| 6 | Scorecard her job için yazılır, mevcut manifest'i bozmaz | ✅ Atomic write, paralel dosya |
| 7 | Mevcut testlerde yeni regresyon yok | ✅ 1 pre-existing fail aynı, 0 yeni |
| 8 | Smoke check Sprint 1 modüllerini doğruluyor | ✅ Stage 2 import edildi |

## Mimari Sınırlar (Sprint 1 Kapsamında Olmayan)

- ❌ Validator henüz scorecard'ı **karar girdisi** olarak kullanmıyor — Sprint 2 işi
- ❌ UI scorecard'ı render etmiyor — Sprint 6 (GS review) ile birlikte planlanacak
- ❌ Auto-preset recommendation scorecard trend'i okumuyor — Sprint 8 işi
- ❌ Coverage metrics henüz upload-zamanı gate yapmıyor (post-mortem only) — Sprint 2

## Sıradaki: Sprint 2 — Capture Quality Gate v2

Coverage metric'leri artık ölçülüyor; Sprint 2 bunları **upload-zamanı gate**'e dönüştürecek:
- Multi-height bucket → guidance overlay
- Azimuth gap > threshold → re-shoot warning
- Burst blur detection
- Operator-facing 3×8 capture matrix UI overlay

Hazır olduğunda başlatabilirim.
