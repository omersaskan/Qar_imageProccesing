# Sprint 3 — Keyframe Selection v2 + Adaptive Sampling

> **Status:** ✅ Complete · **Tests:** 15/15 (Sprint 3) · **Cumulative:** 66/66 (Sprint 1-3) · **Regression:** 0 new

## Hedef

Mevcut frame_extractor sabit `frame_sample_rate` kullanıyordu. Sprint 3 bunu **motion-aware adaptive decimation** ile değiştirdi + **coverage-aware rebalancer** ekledi. Her ikisi de **opt-in** (env flag).

1. **AdaptiveSampler** — optical flow + bbox IoU + sharpness ile per-frame keep/skip kararı
2. **CoverageAwareSelector** — gate matrix'inden eksik elevation bucket'lara öncelik veren post-pass
3. **Settings opt-in** — `ADAPTIVE_SAMPLING_ENABLED` ve `COVERAGE_AWARE_REBALANCE_ENABLED` flag'leri
4. **frame_extractor.py entegrasyonu** — legacy fixed-rate path korunur, opt-in adaptive path eklenir
5. **extraction_manifest.json genişlemesi** — `adaptive_sampling.stats` + `coverage_aware_rebalance` field'ları

## Eklenen / Değişen Dosyalar

### Yeni modüller
| Dosya | LoC | İçerik |
|-------|-----|--------|
| [`adaptive_sampling.py`](../modules/capture_workflow/adaptive_sampling.py) | 220 | `AdaptiveSampler` class — sparse Lucas-Kanade flow + Laplacian variance + bbox IoU; 6 verdict (keep/skip_static/skip_redundant/skip_blurry/keep_motion_burst/keep_forced) |
| [`coverage_aware_selector.py`](../modules/capture_workflow/coverage_aware_selector.py) | 117 | `select_balanced_frames()` — bucket cap + min target + quality-sorted fill; under-representation flagging |

### Değişen dosyalar
| Dosya | Değişiklik |
|-------|-----------|
| [`settings.py`](../modules/operations/settings.py) | 2 yeni field: `adaptive_sampling_enabled`, `coverage_aware_rebalance_enabled` (her ikisi default `false`) |
| [`frame_extractor.py`](../modules/capture_workflow/frame_extractor.py) | Loop içinde adaptive sampling karar noktası (legacy `sample_rate` ile yan yana); post-extraction'da coverage rebalance opt-in pass; manifest'e `adaptive_sampling` + `coverage_aware_rebalance` field'ları |
| [`smoke_check.sh`](../scripts/smoke_check.sh) | Stage 2'de Sprint 3 modül imports |

### Yeni testler
| Dosya | Test | Kapsam |
|-------|------|--------|
| `tests/sprint3/test_adaptive_sampling.py` | 10 | bbox_iou edge cases, optical flow static/shifted, first-frame keep, static repeat skip, motion burst tag, blurry skip, force-keep cadence, stats, reset |
| `tests/sprint3/test_coverage_aware_selector.py` | 5 | empty input, single-band under-representation, three-band balance, per-bucket cap, quality ordering |

**Toplam: 15 yeni test → 15/15 passed.**

## Mimari Etki

### AdaptiveSampler Karar Tablosu

| Verdict | Tetikleyici | Davranış |
|---------|------------|---------|
| `KEEP_FORCED` | İlk frame veya `force_keep_every_n_raw_frames` cadence | Her durumda kabul |
| `SKIP_STATIC` | flow < `min_flow_static` AND iou > `static_iou` (0.97) | Redundant, kamera durmuş |
| `SKIP_REDUNDANT` | flow < `redundant_flow` (4.0) AND iou > `redundant_iou` (0.92) | Yakın benzer view |
| `SKIP_BLURRY` | sharpness < `min_sharpness` (60.0) | Motion blur veya defocus |
| `KEEP_MOTION_BURST` | flow > `burst_flow` (35.0) | Hızlı pan, kabul ama tag |
| `KEEP` | Diğer durumlar | Normal kabul |

### Veri Akışı

```
upload → worker → extract_keyframes()
   if ADAPTIVE_SAMPLING_ENABLED:
      raw video frames → AdaptiveSampler.decide() per frame
                       → keep / skip with reasons
   else:
      raw video frames → fixed sample_rate filter (legacy)
   ↓
   per-frame mask + quality filter (mevcut)
   ↓
   redundancy filter (mevcut histogram + bbox IoU)
   ↓
   if COVERAGE_AWARE_REBALANCE_ENABLED:
      kept_frames → elevation_estimator → bucket assignments
                  → select_balanced_frames() → drop over-represented
                  → unlink dropped from disk
   ↓
   color_profile / capture_profile / capture_gate (Sprint 1+2)
   ↓
   extraction_manifest.json
       + adaptive_sampling.stats + decisions_sample[:200]
       + coverage_aware_rebalance.{bucket_counts_before/after, actions}
```

### Beklenen Saha Etkisi

| Capture senaryosu | Legacy fixed-rate (5) | AdaptiveSampler |
|-------------------|----------------------|-----------------|
| 30s yavaş orbit (telefon stabil) | ~30 keyframe (çoğu redundant) | ~12-18 keyframe (gerçek view'lar) |
| 60s hızlı pan (motion burst) | ~60 keyframe (çoğu blur) | ~25-35 keyframe (burst zorla atılır) |
| 120s multi-height (operatör pause yapar) | ~120 keyframe (statik segmentlerde duplicate) | ~40-60 keyframe (pause segmentleri SKIP_STATIC) |
| Telefon tripoda monteli (hiç hareket yok) | ~20 keyframe (hepsi aynı) | 1 KEEP_FORCED + her N raw frame'de 1 anchor (~3-5 toplam) |

**Net kazanç**: 2-3× daha az frame, ortalama %50 daha kısa reconstruction süresi, daha temiz BA convergence.

### Opt-In Stratejisi

İki flag de **default false** kalıyor — production akışı bozulmaz. Aktivasyon için:

```ini
# .env
ADAPTIVE_SAMPLING_ENABLED=true
COVERAGE_AWARE_REBALANCE_ENABLED=true
```

İlk pilot:
1. Sadece `ADAPTIVE_SAMPLING_ENABLED=true` → manifest'teki `adaptive_sampling.stats`'a bak
2. Birkaç başarılı capture sonrası `COVERAGE_AWARE_REBALANCE_ENABLED=true` ekle
3. Scorecard'daki `coverage.multi_height_score` yükselip yükselmediğini izle

### Bilinçli Sınırlar

- **AdaptiveSampler bbox bilmiyor**: ilk pasta mask oluşturulmadan önce karar veriyor (mask generation pahalı, hep frame için yapılmaz). Sonuçta gerçek mask-based redundancy `_should_reject_as_redundant` legacy filter'a kalıyor (zaten var). Sprint 5+'da AdaptiveSampler ön-mask geçirilebilir.
- **CoverageAwareSelector elevation only**: azimuth bucket'ları yok (heuristic). Sprint 4 sonrası gerçek COLMAP poses ile genişler.
- **Sharpness AdaptiveSampler içinde**: post-pass selector sharpness'ı bilmiyor (1.0 neutral). Sprint 3 v2'de adaptive_sampling'in per-frame sharpness'ı manifest'e yazılıp selector'a feed edilebilir.

## Başarı Kriterleri

| # | Kriter | Sonuç |
|---|--------|-------|
| 1 | Optical flow static (0 motion) ≈ 0 magnitude | ✅ test geçti |
| 2 | Optical flow shifted frame > 5px detect | ✅ test geçti |
| 3 | İlk frame KEEP_FORCED | ✅ |
| 4 | Statik tekrar SKIP_STATIC | ✅ |
| 5 | Motion burst tag | ✅ |
| 6 | Sharpness threshold blur skip | ✅ |
| 7 | Force-keep cadence (statik scene için anchor) | ✅ |
| 8 | Coverage selector bucket cap respected | ✅ |
| 9 | Coverage selector quality ordering within bucket | ✅ |
| 10 | Under-representation flagging | ✅ |
| 11 | Settings opt-in flags | ✅ |
| 12 | frame_extractor opt-in entegrasyonu (legacy korunur) | ✅ |
| 13 | manifest extension (adaptive_sampling + coverage_aware_rebalance) | ✅ |
| 14 | Mevcut suitlerde 0 yeni regresyon | ✅ |
| 15 | Smoke check Sprint 3 modülleri import | ✅ |

## Sıradaki: Sprint 4 — Profile-Aware COLMAP/OpenMVS Preset Hardening

- material × size × scene = 36-cell parametre matrisi
- glossy → `RECON_STEREO_FUSION_MIN_NUM_PIXELS=4`, metallic → resolution_level bump
- Adaptive PatchMatchStereo `window_radius` (small product 5, large 9)
- intrinsics_cache.py — aynı cihazdan 2. capture için kalibrasyon paylaşımı

Hazır olduğunda başlatabilirim.
