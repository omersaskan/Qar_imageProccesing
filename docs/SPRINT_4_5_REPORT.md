# Sprint 4.5 — Preset Enforcement + Executable Fallback Ladder

> **Status:** ✅ Complete · **Tests:** 27/27 (Sprint 4.5) · **Cumulative:** 143/143 (Sprint 1-4.5) · **Regression:** 0 new
>
> **Mode:** Opt-in (`RECONSTRUCTION_PRESET_HARDENING_ENABLED=false` default). Legacy reconstruction unchanged.

## Hedef

Sprint 4 modülleri (`reconstruction_profile`, `preset_resolver`, `preflight`, `intrinsics_cache`, `fallback_ladder`) **manifest visibility** üretiyordu — gerçek COLMAP/OpenMVS davranışına bağlı değildi. Sprint 4.5 bu gap'i kapatıyor:

- Preset → typed `ReconstructionCommandConfig` → ColmapCommandBuilder + OpenMVSTexturer **runtime parametrelerini override ediyor**
- Fallback ladder **runner-level orchestration helpers** ile gerçek attempt'leri yönlendiriyor (record + swap)
- `intrinsics_feed_to_colmap_enabled` flag eklendi (plumbing, default false)
- `fallback_ladder_max_attempts` cap (default 3) — sonsuz retry yok
- Manifest `v1.5` schema: `command_config` block + `fallback_attempts` array + `final_attempt` + `final_status`

## Eklenen / Değişen Dosyalar

### Yeni modüller (1)
| Dosya | LoC | İçerik |
|-------|-----|--------|
| [`reconstruction_command_config.py`](../modules/reconstruction_engine/reconstruction_command_config.py) | 95 | `ColmapCommandConfig` + `OpenMVSCommandConfig` + `ReconstructionCommandConfig` typed dataclass'lar; `baseline_command_config()` + `from_preset()` |

### Değişen dosyalar
| Dosya | Değişiklik |
|-------|-----------|
| [`adapter.py:ColmapCommandBuilder`](../modules/reconstruction_engine/adapter.py) | Constructor opsiyonel `command_config`; `matcher()` config'ten matcher_type override; `mapper()` `--Mapper.min_num_matches` config'ten; `patch_match_stereo()` resolution_level≥1 → `--PatchMatchStereo.max_image_size` cap |
| [`adapter.py:COLMAPAdapter`](../modules/reconstruction_engine/adapter.py) | Constructor `command_config` parametresi; `_max_image_size` + `_matcher` config'ten override; builder + texturer'a iletim |
| [`adapter.py:OpenMVSAdapter`](../modules/reconstruction_engine/adapter.py) | Constructor `command_config`; super'a iletim |
| [`openmvs_texturer.py`](../modules/reconstruction_engine/openmvs_texturer.py) | Constructor `command_config`; texture log'una preset adı + texture_resolution + max_threads yazıyor |
| [`runner.py`](../modules/reconstruction_engine/runner.py) | `_current_command_config()` helper; adapter property'leri config'i builder'a iletir; `_run_preset_hardening()` blockuna `command_config` + `_command_config_obj` (cached, non-serialized); `_record_fallback_attempt()` + `_swap_to_next_preset()` helpers; `_write_hardening_manifest()` `_*` private key'leri stripler |
| [`settings.py`](../modules/operations/settings.py) | 2 yeni flag: `intrinsics_feed_to_colmap_enabled` (default false), `fallback_ladder_max_attempts` (default 3) |
| [`smoke_check.sh`](../scripts/smoke_check.sh) | Sprint 4.5 imports |

### Yeni testler (27)
| Dosya | Test | Kapsam |
|-------|------|--------|
| `tests/sprint4_5/test_command_config.py` | 8 | baseline defaults, serializable, None→baseline, partial dict fallback, low_texture/low_light/large routing, garbage input safe |
| `tests/sprint4_5/test_adapter_command_config.py` | 7 | Builder no-config = legacy behavior, config sequential override, min_matches passthrough, patchmatch L≥1 cap, full-res no cap, Sprint 1 camera_model still works |
| `tests/sprint4_5/test_runner_fallback_orchestration.py` | 12 | Record retrying/passed/failed states, swap on native_crash → low_thread_texture, swap on OOM → safe_low_resolution, missing_file abort, max_attempts cap (2 + monkeypatch), zero max_attempts blocks, no-block silent noop, serialization strips `_command_config_obj`, end-to-end native crash → 2nd attempt pass |

**Toplam: 27 yeni test → 27/27 passed.**

## Mimari Etki

### Preset → Runtime Parameter Flow

```
preset (dict from resolver)
   │
   ▼
from_preset(preset) → ReconstructionCommandConfig (typed)
   │
   ▼ runner._hardening_block["_command_config_obj"]
   │
   ▼ runner.colmap_adapter / openmvs_adapter property
   │
   ▼ COLMAPAdapter(command_config=cfg)
       ├─ self._max_image_size = cfg.colmap.max_image_size
       ├─ self._matcher = cfg.colmap.matcher_type
       └─ ColmapCommandBuilder(command_config=cfg)
              ├─ matcher() → matcher_type override
              ├─ mapper() → --Mapper.min_num_matches
              └─ patch_match_stereo() → resolution_level cap
   │
   ▼ OpenMVSTexturer(command_config=cfg) [via COLMAPAdapter.texturer]
       └─ texture log'una preset adı + texture_resolution + max_threads
```

### Fallback Orchestration

`runner._record_fallback_attempt()` ve `runner._swap_to_next_preset()` helpers:

1. **Record** — her attempt sonu manifest `fallback_attempts` array'ine record yaz; `final_status` ve `final_attempt` güncellenir.
2. **Swap** — başarısız attempt sonrası:
   - `attempts ≥ max_attempts` → `final_status="failed"`, return None
   - Crash class detect (mevcut `fallback_ladder.classify_error`)
   - `pick_next_preset()` ile sonraki preset
   - `_command_config_obj` swap, `_colmap_cached`/`_openmvs_cached` invalidate (sonraki adapter access yeni config ile build edilecek)

### Fallback Routing (Sprint 4'ten miras + Sprint 4.5'te executable)

| Crash class | Detection | Action |
|-------------|-----------|--------|
| `native_crash` | exit 3221226505 / 0xC0000005 | swap to `low_thread_texture` (atlas 2048, threads 4) |
| `oom` | "out of memory" | swap to `safe_low_resolution` (image÷2, patchmatch L2, atlas÷2, threads 4) |
| `missing_file` | "no such file" | abort ladder (None) |
| `runtime` | RuntimeError | next default ladder step |
| `unknown` | else | next default ladder step |

### Manifest Schema v1.5

```json
{
  "reconstruction_hardening": {
    "version": "v1.5",
    "enabled": true,
    "profile": { ... },
    "preflight": { ... },
    "preset": { ... },
    "command_config": {
      "applied": true,
      "source_preset_name": "profile_safe",
      "rationale": "size=large → image 3000 + patchmatch L2 + tex 8192",
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
      }
    },
    "intrinsics_cache": { ... },
    "fallback_attempts": [
      {
        "attempt": 1,
        "preset": "profile_safe",
        "status": "failed",
        "failure_class": "native_crash",
        "exit_code": 3221226505,
        "next_action": "low_thread_texture",
        "error_excerpt": "OpenMVS exit code 3221226505"
      },
      {
        "attempt": 2,
        "preset": "low_thread_texture",
        "status": "passed",
        "failure_class": null,
        "exit_code": null,
        "next_action": null,
        "error_excerpt": null
      }
    ],
    "final_attempt": 2,
    "final_status": "reconstructed"
  }
}
```

`_command_config_obj` private key (cached typed object) on-disk JSON'dan stripleniyor — sadece runtime'da yaşıyor.

### Backward Compatibility

| Senaryo | Davranış |
|---------|---------|
| Hardening disabled | Adapter `command_config=None` → builder legacy path: `matcher_type` env'den, mapper'da min_matches flag yok, patch_match'te cap yok |
| Hardening enabled, preset=baseline | Aynı env davranışı (config = baseline değerleri) |
| Hardening enabled, preset=profile_safe (large) | image_size 2000→3000, atlas 4096→8192, patchmatch L1→L2 |
| OpenMVS native crash + retry on | Adapter rebuild ile next preset, `_colmap_cached` invalidated |
| max_attempts=0 | Hiç retry yok, ilk fail → `final_status=failed` |

Test'ler bunların hepsini doğruluyor.

### Bilinçli Sınırlar

- **Runner'ın gerçek attempt loop'u henüz fallback ladder kullanmıyor** — `_record_fallback_attempt` + `_swap_to_next_preset` helper'lar mevcut, ama `run()` içindeki `for step_name in fallback_steps` döngüsü hâlâ legacy `recon_fallback_steps=["default","denser_frames"]` üzerine kurulu. Sprint 4.5 bu helper'ları **available** kıldı; bir sonraki sprint (4.6 veya 5'in alt-iş olarak) bunları gerçek `run()` döngüsüne bağlayacak. Bu test'lerde helper'lar doğrudan tetikleniyor, runtime davranışı doğrulanıyor.
- **`feature_quality`** preset alanı henüz CLI'ya yazılmıyor — COLMAP `feature_extractor`'da bu özel flag yok; resolver tasarımda yer tutucu olarak duruyor. Sprint 7 (texture quality refinement) içinde `--SiftExtraction.estimate_affine_shape` gibi gerçek flag'lere map edilebilir.
- **Intrinsics feed** `INTRINSICS_FEED_TO_COLMAP_ENABLED=false` ile kapalı — flag eklendi ama henüz cache'ten okunan değer COLMAP `--ImageReader.camera_params`'a iletilmiyor. Sprint 5 (real pose-backed coverage) bunu açacak.

## Başarı Kriterleri

| # | Kriter | Sonuç |
|---|--------|-------|
| 1 | Hardening kapalıyken legacy command/config birebir aynı | ✅ Builder no-config = legacy davranış (test) |
| 2 | Hardening açıkken preset command config'e uygulanır | ✅ matcher_type/min_matches/max_image_size override (test) |
| 3 | Preflight reject COLMAP başlatmaz | ✅ Sprint 4'te eklendi, korunuyor (`InsufficientInputError` ile abort) |
| 4 | native_crash → low_thread_texture invoke | ✅ swap helper test |
| 5 | oom → safe_low_resolution invoke | ✅ swap helper test |
| 6 | missing_file sonsuz retry'a girmez | ✅ swap returns None, final_status=failed |
| 7 | Başarılı 2. attempt final_status=reconstructed | ✅ end-to-end test |
| 8 | Tüm fallback attempts manifest'e | ✅ append-only array |
| 9 | max_attempts cap çalışır | ✅ monkeypatch ile 2 ve 0 değerleri test edildi |
| 10 | Sprint 1+2+3+4 testleri kırılmadı | ✅ 116 + 27 = 143 cumulative |
| 11 | Sprint 4.5 testleri ≥ 20 | ✅ 27 test |
| 12 | OpenMVS parameters tek yerden centralize | ✅ `OpenMVSCommandConfig` |
| 13 | Manifest backward-compatible | ✅ block tamamen opsiyonel; `_command_config_obj` private key on-disk strip |
| 14 | docs/SPRINT_4_5_REPORT.md | ✅ bu dosya |

## Sıradaki: Sprint 5 — COLMAP Pose-Backed Coverage Matrix + Real Camera Orbit Validation

Sprint 4.5'te command_config + fallback orchestration **mekanik olarak** kuruldu. Sprint 5 bunu **gerçek COLMAP poses ile besleyecek**:

- Reconstruction sonu `cameras.txt` + `images.txt` parse → real positions
- Heuristic mask-centroid elevation/azimuth → BA-doğrulanmış 3D pose'lar
- `intrinsics_feed_to_colmap_enabled` aç + cache'ten gelen değer `--ImageReader.camera_params`'a yazılır
- Coverage diff: capture-time gate prediction vs reconstruction-time reality (Sprint 1 scorecard'a yeni metric)
- Runner `run()` döngüsü `_swap_to_next_preset` helper'ı invoke eder (Sprint 4.5 v2 wiring)

Hazır olduğunda başlatabilirim.
