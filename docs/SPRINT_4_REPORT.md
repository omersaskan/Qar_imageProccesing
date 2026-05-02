# Sprint 4 — Reconstruction Preset Hardening + Preflight

> **Status:** ✅ Complete · **Tests:** 50/50 (Sprint 4) · **Cumulative:** 116/116 (Sprint 1-4) · **Regression:** 0 new
>
> **Mode:** Opt-in (`RECONSTRUCTION_PRESET_HARDENING_ENABLED=false` default). Legacy reconstruction unchanged.

## Hedef

Sprint 1-3'te ürettiğimiz capture/coverage sinyalleri (`extraction_manifest`, `capture_gate`, `adaptive_sampling`) reconstruction adapter'ına geçirilmiyor­du. Sprint 4 bu boşluğu kapatıyor: keyframe set'i COLMAP'a göndermeden önce **profile çıkar**, **preflight'tan geçir**, **profile-aware preset seç**, **OpenMVS crash retry için fallback ladder hazırla**.

**Tasarım prensibi**: Hardening default kapalı. Açıldığında bile ilk versiyonda tutucu — preset deltaları küçük, baseline her zaman fallback olarak duruyor.

## Eklenen / Değişen Dosyalar

### Yeni modüller (5)
| Dosya | LoC | İçerik |
|-------|-----|--------|
| [`reconstruction_profile.py`](../modules/reconstruction_engine/reconstruction_profile.py) | 220 | 4-axis classification: material × size × scene × motion. Deterministic. Confidence score. |
| [`reconstruction_preset_resolver.py`](../modules/reconstruction_engine/reconstruction_preset_resolver.py) | 178 | 5 preset (baseline, profile_safe, low_texture_safe, low_light_safe, texture_retry_safe). Routing precedence: material > scene > general profile > baseline. |
| [`reconstruction_preflight.py`](../modules/reconstruction_engine/reconstruction_preflight.py) | 200 | 6-check gate (frame count, dimension consistency, coverage, blur, static_run, missing files). pass / review / reject decision. |
| [`intrinsics_cache.py`](../modules/reconstruction_engine/intrinsics_cache.py) | 195 | File-backed atomic cache, device + WxH + focal-bin key, default seed on miss. |
| [`fallback_ladder.py`](../modules/reconstruction_engine/fallback_ladder.py) | 175 | Crash-class detection (native_crash / oom / missing_file / runtime / unknown) + deterministic 5-step ladder. |

### Değişen dosyalar
| Dosya | Değişiklik |
|-------|-----------|
| [`settings.py`](../modules/operations/settings.py) | 2 yeni flag: `reconstruction_preset_hardening_enabled` + `intrinsics_cache_enabled` (her ikisi default `false`) |
| [`runner.py`](../modules/reconstruction_engine/runner.py) | `_run_preset_hardening(job)` + `_write_hardening_manifest()` yardımcıları; `run()` başlangıcında opt-in çağrı; preflight reject → `capture_quality_rejected` final status (RuntimeError değil); manifest'e `reconstruction_hardening` block embed |
| [`smoke_check.sh`](../scripts/smoke_check.sh) | Stage 2'ye Sprint 4 modül imports |

### Yeni testler (50)
| Dosya | Test | Kapsam |
|-------|------|--------|
| `tests/sprint4/test_reconstruction_profile.py` | 12 | Empty inputs, hint→material mapping, color→scene mapping, motion derivation (4 senaryo), confidence accumulation, JSON serialization |
| `tests/sprint4/test_preset_resolver.py` | 14 | 5 preset adlarının routing'i, large→big atlas, small→küçük atlas, glossy→keep res_level, fast_motion→sequential, get_by_name fallback, schema completeness |
| `tests/sprint4/test_preflight_intrinsics_fallback.py` | 24 | Empty/below-min/below-review preflight, blur thresholds, dimension mismatch, cache key determinism + binning, miss→hit lifecycle, atomic write, corrupt file recovery, crash classification (4 sınıf), ladder traversal, exhaustion |

**Toplam: 50 yeni test → 50/50 passed.**

## Mimari Etki

### Profile Classification Tablosu

| Eksen | Sınıflar | Karar Sinyali |
|-------|----------|---------------|
| **material** | matte / glossy / transparent_reflective / low_texture / unknown | `capture_profile.material_hint` + `color_profile.category` |
| **size** | small / medium / large / unknown | `capture_profile.size_class` (direct map) |
| **scene** | controlled / cluttered / low_light / unknown | `color_profile.product_rgb mean` + `capture_gate.decision/multi_height/blur` |
| **motion** | stable_orbit / fast_motion / static_poor / uneven / unknown | `capture_gate.azimuth.{orbit_progress, static_run}` + `blur.burst_ratio` + `adaptive_sampling.stats` |

### Preset Routing Precedence

```
material ∈ {transparent_reflective, low_texture}
    → low_texture_safe (sequential matcher, min_matches=8, full-res patchmatch)
   else if scene == low_light
    → low_light_safe (image≤1600, sequential, half-res patchmatch, atlas 2048)
   else if any signal known
    → profile_safe (size-aware image+atlas, glossy→keep res_level, fast→sequential)
   else
    → baseline
```

### Preflight Karar Tablosu

| Sinyal | Hard reject | Soft review |
|--------|------------|-------------|
| selected_count | < 3 | < 8 |
| dimension_mismatch | > 20% | — |
| missing/unreadable | ≥ N/3 sample | — |
| coverage_ratio | < 0.10 | < 0.30 |
| median_blur | < 10 | < 30 |
| static_run_ratio | > 85% | > 50% |

`reject` durumunda runner `InsufficientInputError` raise eder + audit `final_status="capture_quality_rejected"` yazılır (RuntimeError olarak değil — capture sorunu, sistem sorunu değil).

### Fallback Ladder

Default 5-step:
```
[0] profile_safe       (initial)
[1] safe_high_quality  (clamp threads=8, otherwise profile_safe)
[2] safe_low_resolution (image÷2, patchmatch L2, atlas÷2, threads=4)
[3] low_thread_texture  (atlas=2048, threads=4, no further retry)
[4] baseline            (env defaults)
```

Crash-class shortcut:
- `exit code 3221226505` → jump to `low_thread_texture` (OpenMVS native crash)
- `out of memory` / `OOM` → jump to `safe_low_resolution`
- `no such file` → ladder abort (`None` returned, recovery imkansız)

### `reconstruction_hardening` Manifest Block (örnek)

```json
{
  "reconstruction_hardening": {
    "version": "v1",
    "enabled": true,
    "profile": {
      "material_profile": "glossy",
      "size_profile": "large",
      "scene_profile": "controlled",
      "motion_profile": "stable_orbit",
      "confidence": 1.0,
      "signals_used": ["material", "size", "scene", "motion"],
      "reasons": [...]
    },
    "preflight": {
      "decision": "pass",
      "selected_count": 24,
      "coverage_ratio": 0.71,
      "median_blur_score": 187.3,
      "static_run_ratio": 0.08,
      "dimension_mismatch_ratio": 0.0,
      "reasons": [],
      "suggestions": []
    },
    "preset": {
      "name": "profile_safe",
      "colmap": {
        "feature_quality": "high",
        "matcher_type": "exhaustive",
        "max_image_size": 3000,
        "mapper_min_num_matches": 15,
        "patchmatch_resolution_level": 2
      },
      "openmvs": {
        "texture_resolution": 8192,
        "max_threads": 0,
        "enable_texture_retry": true
      },
      "rationale": "base from baseline; size=large → image 3000 + patchmatch L2 + tex 8192; material=glossy → keep resolution_level≥1 for stability"
    },
    "intrinsics_cache": {
      "status": "miss",
      "cache_key": "iphone_15_pro|1920x1080|f6.0|any",
      "source": "default"
    },
    "fallback_attempts": []
  }
}
```

Ek olarak runner job_dir'a `reconstruction_hardening.json` standalone dosya yazıyor (manifest yedeği).

## Başarı Kriterleri

| # | Kriter | Sonuç |
|---|--------|-------|
| 1 | Hardening default kapalı; legacy davranış aynen | ✅ Settings flag false; runner if-block bypass |
| 2 | Empty selected_keyframes preflight reject | ✅ |
| 3 | <3 frames preflight reject | ✅ |
| 4 | <8 frames preflight review | ✅ |
| 5 | Dimension mismatch reject | ✅ |
| 6 | Low coverage reject/review tier'lar | ✅ |
| 7 | Low/very-low blur median tier'ları | ✅ |
| 8 | Static run ratio threshold'ları | ✅ |
| 9 | Profile derivation deterministic | ✅ (signals & reasons audit) |
| 10 | Low-texture material → low_texture_safe preset | ✅ |
| 11 | Low-light scene → low_light_safe preset | ✅ |
| 12 | Unknown profile → baseline preset | ✅ |
| 13 | Large size → bigger image + atlas | ✅ |
| 14 | Glossy material keeps PatchMatch resolution_level ≥ 1 | ✅ |
| 15 | Fast motion → sequential matcher | ✅ |
| 16 | All preset schemas have required keys | ✅ |
| 17 | Intrinsics cache key determinism + focal binning | ✅ |
| 18 | Cache miss→hit lifecycle | ✅ |
| 19 | Cache atomic write + corrupt-file recovery | ✅ |
| 20 | Disabled lookup returns default | ✅ |
| 21 | Crash classifier 4 classes (native/oom/missing/runtime) | ✅ |
| 22 | Native crash → low_thread_texture jump | ✅ |
| 23 | OOM → safe_low_resolution jump | ✅ |
| 24 | Missing file → ladder abort (None) | ✅ |
| 25 | Default ladder ends with baseline | ✅ |
| 26 | Ladder exhaustion returns None | ✅ |
| 27 | Preflight reject → audit capture_quality_rejected | ✅ runner integration |
| 28 | Manifest backward-compatible | ✅ block opsiyonel, varsayılan yok |
| 29 | OpenMVS parameters centralized in resolver | ✅ texture_resolution + max_threads + enable_texture_retry tek yerde |
| 30 | Mevcut suite regresyon | ✅ 0 yeni (1 pre-existing aynı) |
| 31 | Tüm Sprint 1-4 testleri geçer | ✅ 116/116 |

## Bilinçli Sınırlar

- **Preset değerleri henüz COLMAP komut satırına yazılmıyor**. Sprint 4 v1 sadece manifest visibility üretiyor — adapter.py'da `feature_quality` / `mapper_min_num_matches` parametre propagation Sprint 4 v2 işi (riskli, daha geniş test gerek).
- **Runner fallback ladder'ı henüz invoke etmiyor**. Mevcut `recon_fallback_steps=["default","denser_frames"]` legacy sırası korunuyor. Ladder sadece preset suggestion üretici. Runner-level retry orchestration Sprint 4 v3.
- **Intrinsics cache sadece probe** — değer COLMAP'a `--ImageReader.camera_params` olarak henüz iletilmiyor. Sprint 5 (gerçek COLMAP pose-backed coverage) ile birlikte feed edilecek.
- **Material classification kaba**: `glossy` ve `low_texture` capture_profile.material_hint'ten geliyor, gerçek görüntü analizi yok. Sprint 7 (texture quality refinement) ile contrast/highlight analizi eklenir.
- **Scene `cluttered` henüz hiç bir kondisyon altında üretilmiyor** — derivation logic'i hazır ama background dominance metriği Sprint 7 işi.
- **Confidence score sadece signal-count, kalite-ağırlıklı değil**.

## Opt-in Aktivasyon

```ini
# .env
RECONSTRUCTION_PRESET_HARDENING_ENABLED=true
INTRINSICS_CACHE_ENABLED=true
```

Pilot iş akışı:
1. Hardening aç → bir job çalıştır → `manifest.json.reconstruction_hardening`'e bak
2. Profile + preset doğru mu kontrol et → preset değerleri loglanmış (rationale field)
3. Preflight reject olursa: `audit.json.final_status == "capture_quality_rejected"` görünür, kullanıcı re-shoot guidance alır
4. Birkaç başarılı job sonrası `INTRINSICS_CACHE_ENABLED=true` → `data/intrinsics_cache.json` doluyor

## Sıradaki: Sprint 5 — COLMAP Pose-Backed Coverage Matrix + Real Camera Orbit Validation

Sprint 1-4 boyunca capture-time heuristic kullandık (mask centroid bazlı pseudo-elevation/azimuth). Sprint 5 bunu **gerçek COLMAP poses** ile değiştirecek:

- **Reconstruction sonu** `cameras.txt` + `images.txt` parse → gerçek camera positions
- **Real azimuth/elevation buckets** — heuristic değil, BA-doğrulanmış
- **Coverage diff** — capture-time gate prediction vs reconstruction-time reality
- **Intrinsics feedback** — BA çıkışı `intrinsics_cache`'e yazılır (sonraki capture için kalibrasyon prior'u olur)
- **Preset propagation v2** — preset değerleri gerçekten COLMAP/OpenMVS komut satırına yazılır (adapter integration)

Hazır olduğunda başlatabilirim.
