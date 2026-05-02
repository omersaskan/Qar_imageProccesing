# Sprint 4.6 — Runtime Fallback Loop Wiring + Attempt Orchestration

> **Status:** ✅ Complete · **Tests:** 40/40 (Sprint 4.6) · **Cumulative:** 183/183 (Sprint 1-4.6) · **Regression:** 0 new
>
> **Mode:** Opt-in. Default off (`RECONSTRUCTION_PRESET_HARDENING_ENABLED=false`,
> `RECONSTRUCTION_RUNTIME_FALLBACK_ENABLED=false`). Legacy reconstruction unchanged.

## Hedef

Sprint 4.5'te eklenen `command_config` abstraction + `fallback_ladder` helper'ları + preset swap mekanizması **runner içinde duran ama tetiklenmeyen** yardımcılardı. Sprint 4.6 onları gerçek `runner.run()` retry loop'una bağlıyor:

- Hardening + runtime fallback **birlikte** açıkken legacy `recon_fallback_steps` döngüsü yerine preset-aware ladder driver'ı çalışıyor.
- Her attempt manifest'e zengin metadata ile yazılıyor (preset, command_config snapshot, started/finished timestamps, failure_class, exit_code, error_summary, next_preset).
- Native crash / OOM / missing_file / timeout / unknown sınıfları gerçek exception'lardan tespit ediliyor.
- `missing_file` sonsuz retry'a girmiyor: deterministik abort.
- `fallback_ladder_max_attempts` cap (Sprint 4.5'ten) gerçek retry loop'unda enforce ediliyor.

## Eklenen / Değişen Dosyalar

### Değişen dosyalar
| Dosya | Değişiklik |
|-------|-----------|
| [`runner.py`](../modules/reconstruction_engine/runner.py) | `_classify_attempt_failure()` static helper; `_runtime_fallback_active()` + `_run_runtime_fallback_loop()` + `_peek_next_preset()`; `_record_fallback_attempt()` extended with Sprint 4.6 fields (attempt_index, command_config, started_at/finished_at, error_summary, next_preset); `_swap_to_next_preset()` updates `active_preset`; `_run_preset_hardening()` bumped to v1.6 with hardening_mode + runtime_fallback_enabled + active_preset + attempts[]; `run()` dispatches to runtime loop when active and mirrors `final_status="capture_quality_rejected"` into hardening block on preflight reject. |
| [`settings.py`](../modules/operations/settings.py) | New flag `reconstruction_runtime_fallback_enabled` (default false, env `RECONSTRUCTION_RUNTIME_FALLBACK_ENABLED`). |

### Yeni testler (40)
| Dosya | Test count | Kapsam |
|-------|------------|--------|
| `tests/sprint4_6/test_failure_classification.py` | 10 | native_crash (exit code 3221226505 / 0xC0000005 / TexturingFailed), oom ("out of memory" / "memory allocation"), missing_file (string match + MissingArtifactError), timeout, unknown default, summary truncation. |
| `tests/sprint4_6/test_runtime_fallback_loop.py` | 30 | First-attempt initial preset; native_crash → low_thread_texture retry; oom → safe_low_resolution retry; missing_file abort (no retry); unknown → default ladder; max_attempts cap; all-fail → final_status=failed; zero max_attempts → 1 attempt; manifest v1.6 fields per record; deterministic order; serialization strips private keys; v1.6 mode reflects flag (manifest_only vs runtime_enforced); adapter cache invalidation; run() dispatch routing on/off; hardening flag off → block None; preflight reject writes capture_quality_rejected to both audit and hardening manifest, adapter never called; audit attempts appended; attempt directories created; persistent native_crash never infinite-retries; command_config snapshot changes per preset; timestamps recorded; attempts alias mirrors fallback_attempts; record sets active_preset on pass; runtime active gate. |

**Toplam: 40 yeni test → 40/40 passed.**

## Mimari Etki

### Runtime dispatch matrisi

| `reconstruction_preset_hardening_enabled` | `reconstruction_runtime_fallback_enabled` | Loop | `hardening_mode` |
|---|---|---|---|
| false | * | Legacy (`recon_fallback_steps`) | n/a (block not built) |
| true | false | Legacy (`recon_fallback_steps`) — adapters still see preset command_config (Sprint 4.5) | `manifest_only` |
| true | true | **Sprint 4.6 preset-aware ladder** | `runtime_enforced` |

### Preset-aware retry loop (`_run_runtime_fallback_loop`)

```
┌───────────────────────────────────────────────────────────────┐
│ attempt_num = 0                                               │
│ while attempt_num < fallback_ladder_max_attempts:             │
│   attempt_num += 1                                            │
│   preset = block.preset.name                                  │
│   cfg_snapshot = block._command_config_obj.to_dict()          │
│   start = utcnow()                                            │
│   try:                                                        │
│     results = self.adapter.run_reconstruction(...)            │
│     audit.attempts += [success]                               │
│     _record_fallback_attempt(passed, …)                       │
│     return results                                            │
│   except Exception as exc:                                    │
│     class, exit_code, summary = _classify_attempt_failure(exc)│
│     audit.attempts += [failed]                                │
│     if class == "missing_file":                               │
│       _record_fallback_attempt(failed, next_preset=None)      │
│       return None  # deterministic abort                      │
│     next_preset = _peek_next_preset(summary)                  │
│     if attempts+1 >= max_attempts: next_preset = None         │
│     _record_fallback_attempt(failed, next_preset, …)          │
│     if next_preset is None: return None                       │
│     _swap_to_next_preset(summary)  # rebuilds adapter         │
│ return None  # exhausted                                      │
└───────────────────────────────────────────────────────────────┘
```

### Failure classification (`_classify_attempt_failure`)

| Class | Detection (case-insensitive) |
|-------|------------------------------|
| `native_crash` | `"3221226505"` / `"0xC0000005"` / `"TEXTUREMESH_NATIVE_CRASH"` / `"native crash"` / `isinstance(exc, TexturingFailed)`; exit_code defaults to 3221226505 if otherwise unknown |
| `oom` | `"out of memory"` / `"memory allocation"` / `"cuda error: out of memory"` / `"oom"` |
| `missing_file` | `"no such file"` / `"file not found"` / `"missing artifact"` / `isinstance(exc, MissingArtifactError)` |
| `timeout` | `"timeout"` / `"timed out"` |
| `unknown` | else |

### Manifest schema v1.6

```json
{
  "reconstruction_hardening": {
    "version": "v1.6",
    "enabled": true,
    "hardening_mode": "runtime_enforced",
    "runtime_fallback_enabled": true,
    "profile": { ... },
    "preflight": { "decision": "pass" },
    "preset": { ... },
    "active_preset": "low_thread_texture",
    "command_config": { ... },
    "intrinsics_cache": { ... },
    "attempts": [
      {
        "attempt_index": 1,
        "preset_name": "profile_safe",
        "command_config": { "applied": true, "source_preset_name": "profile_safe", "colmap": { ... }, "openmvs": { ... } },
        "started_at": "2026-05-02T12:34:56.789012+00:00",
        "finished_at": "2026-05-02T12:38:11.998765+00:00",
        "status": "failed",
        "failure_class": "native_crash",
        "exit_code": 3221226505,
        "error_summary": "OpenMVS exit code 3221226505 during TextureMesh",
        "next_preset": "low_thread_texture",
        "attempt": 1, "preset": "profile_safe", "next_action": "low_thread_texture", "error_excerpt": "..."
      },
      {
        "attempt_index": 2,
        "preset_name": "low_thread_texture",
        "command_config": { ... },
        "started_at": "...", "finished_at": "...",
        "status": "passed",
        "failure_class": null, "exit_code": null,
        "error_summary": null, "next_preset": null,
        "attempt": 2, "preset": "low_thread_texture", "next_action": null
      }
    ],
    "fallback_attempts": [ /* same records — kept as legacy alias for backward compat */ ],
    "final_attempt": 2,
    "final_status": "reconstructed"
  }
}
```

`_command_config_obj` cached private key on-disk JSON'dan stripleniyor (Sprint 4.5 davranışı korundu).

### Backward compatibility

| Senaryo | Davranış |
|---------|---------|
| Hardening disabled | run() → legacy fallback_steps loop (`["default", "denser_frames"]`); `_hardening_block = None`; manifest'te `reconstruction_hardening` yok. |
| Hardening enabled, runtime fallback disabled | run() → legacy loop; adapter command_config Sprint 4.5'teki gibi preset'ten besleniyor; manifest `hardening_mode="manifest_only"`. |
| Hardening enabled, runtime fallback enabled | run() → preset-aware ladder; manifest v1.6 attempts[] her retry kayıt ediyor. |
| Preflight reject | Hem audit hem hardening manifest `final_status="capture_quality_rejected"`; reconstruction adapter çağrılmıyor. |
| `fallback_ladder_max_attempts=0` | Floor 1: tek attempt; başarısızsa `final_status=failed`. Sonsuz retry yok. |
| Sprint 4.5 helper testleri | Tümü geçerliliğini koruyor — `_record_fallback_attempt` Sprint 4.6 alanlarını additive olarak ekliyor, eski key'leri (attempt, preset, next_action, error_excerpt) silmiyor. |

## Başarı Kriterleri

| # | Kriter | Sonuç |
|---|--------|-------|
| 1 | Hardening kapalıyken legacy fallback loop birebir korunuyor | ✅ test_run_dispatch_skips_runtime_loop_when_hardening_off |
| 2 | Hardening açıkken initial preset attempt 1'de kullanılıyor | ✅ test_runtime_loop_first_attempt_uses_initial_preset |
| 3 | native_crash → low_thread_texture ile retry | ✅ test_runtime_loop_native_crash_retries_on_low_thread_texture |
| 4 | oom → safe_low_resolution ile retry | ✅ test_runtime_loop_oom_retries_on_safe_low_resolution |
| 5 | missing_file → retry yapmadan abort | ✅ test_runtime_loop_missing_file_aborts_without_retry (1 adapter call total) |
| 6 | unknown failure → default ladder sıradaki preset | ✅ test_runtime_loop_unknown_failure_uses_default_ladder_step (safe_high_quality) |
| 7 | max_attempts sınırı çalışır | ✅ test_runtime_loop_max_attempts_cap_enforced (cap=2 → 2 calls) |
| 8 | İkinci attempt başarılı → final_status=reconstructed | ✅ native_crash + oom retry testleri |
| 9 | Tüm attempts başarısız → final_status=failed | ✅ test_runtime_loop_all_attempts_fail_sets_final_failed |
| 10 | Preflight reject → COLMAP/OpenMVS çağrılmaz | ✅ test_run_preflight_reject_writes_capture_quality_rejected_and_skips_recon |
| 11 | Adapter cache invalidation sonrası yeni command_config | ✅ test_swap_preset_invalidates_cached_adapters_and_rebuilds_with_new_config |
| 12 | Manifest attempts sırası deterministik | ✅ test_manifest_v1_6_attempts_order_deterministic |
| 13 | Sprint 1-4.5 testleri kırılmadı | ✅ Cumulative 183/183 |
| 14 | Sprint 4.6 testleri ≥ 25 | ✅ 40 test |
| 15 | Hardening default kapalı | ✅ Her iki flag de default false |
| 16 | Sprint 4.5 helper'ları artık runtime davranışı etkiliyor | ✅ `_run_runtime_fallback_loop` doğrudan `_record_fallback_attempt`, `_swap_to_next_preset`, `_peek_next_preset` (yeni `pick_next_preset` wrapper) çağırıyor |
| 17 | docs/SPRINT_4_6_REPORT.md | ✅ Bu dosya |

## Bilinçli Sınırlar (spec'in tekrarı)

- **COLMAP pose-backed coverage matrix** bu sprintte yapılmadı.
- **`cameras.txt` / `images.txt` parser** yok.
- **Intrinsics feed** (`INTRINSICS_FEED_TO_COLMAP_ENABLED`) hâlâ default kapalı; bu sprintte plumbing değişmedi.
- Sprint 4.6 **sadece runtime fallback enforcement** sprintidir.

## Sıradaki: Sprint 5 — Real Camera Pose-Backed Coverage Validation

Sprint 4.6'da preset-aware ladder gerçek runtime'a bağlandı. Sprint 5 bunu COLMAP poses ile besleyecek (cameras.txt/images.txt parse, intrinsics feed open, capture-time vs reconstruction-time coverage diff).
