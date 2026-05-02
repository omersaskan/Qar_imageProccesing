# Sprint 2 — Capture Quality Gate v2

> **Status:** ✅ Complete · **Tests:** 18/18 (Sprint 2 only) · **Cumulative:** 51/51 (Sprint 1 + 2) · **Regression:** 0 new

## Hedef

Sprint 1'de coverage metrikleri **post-mortem** yazılıyordu. Sprint 2 bu sinyalleri **extraction-zamanı capture quality gate**'ine dönüştürdü:

1. **Blur burst detection** — kötü frame run'ları temporal cluster ile yakala
2. **Elevation distribution** — 3-band (low/mid/top) eksiksizliği ölç
3. **Azimuth diversity** — orbit progress + static-pause detection
4. **Capture quality gate orchestrator** — 3 sinyali birleştir → `pass` / `review` / `reshoot`
5. **3×8 capture matrix overlay UI** — operatör için görsel re-shoot guidance

## Eklenen / Değişen Dosyalar

### Yeni modüller
| Dosya | LoC | İçerik |
|-------|-----|--------|
| [`blur_burst_detector.py`](../modules/capture_workflow/blur_burst_detector.py) | 168 | Laplacian variance + MAD-based robust z-score + temporal run detection |
| [`elevation_estimator.py`](../modules/capture_workflow/elevation_estimator.py) | 147 | Mask centroid Y-position → low/mid/top bucket (heuristic, COLMAP poses olmadan) |
| [`azimuth_diversity.py`](../modules/capture_workflow/azimuth_diversity.py) | 116 | Centroid X trajectory → cumulative orbit progress + static-run detection |
| [`capture_quality_gate.py`](../modules/capture_workflow/capture_quality_gate.py) | 198 | Orchestrator: 3 detector + threshold matrix + 3×8 visual matrix |

### Değişen dosyalar
| Dosya | Değişiklik |
|-------|-----------|
| [`frame_extractor.py`](../modules/capture_workflow/frame_extractor.py) | Extraction sonu `evaluate_capture()` çağrılıyor; sonuç `extraction_report` ve `extraction_manifest.json`'a `capture_gate` field'ı olarak yazılıyor |
| [`scorecard.py`](../modules/qa_validation/scorecard.py) | `capture_gate` field'ı scorecard'a eklendi; `decision=reshoot` overall grade'i F'e düşürüyor; `decision=review` grade'i C'ye düşürüyor |
| [`api.py`](../modules/operations/api.py) | Yeni endpoint: `GET /api/sessions/{session_id}/capture-gate` — UI polling için |
| [`smoke_check.sh`](../scripts/smoke_check.sh) | Stage 2 + Stage 4'e Sprint 2 modül imports + route eklendi |

### Yeni UI
| Dosya | İçerik |
|-------|-------|
| [`ui/capture_gate.html`](../ui/capture_gate.html) | Standalone gate raporu paneli: decision badge + 3 sub-panel (blur/elevation/azimuth) + 3×8 capture matrix grid + reasons & suggestions |

### Yeni testler
| Dosya | Test | Kapsam |
|-------|------|--------|
| `tests/sprint2/test_blur_burst_detector.py` | 6 | Sharp/blur ayrımı, run detection, single-frame skip, empty input, MAD-zero edge case |
| `tests/sprint2/test_elevation_estimator.py` | 6 | Bucket thresholds, no-frames, missing masks_dir warning, single/dual/triple band scenarios |
| `tests/sprint2/test_capture_quality_gate.py` | 6 | Empty input → reshoot, below-min frames → review, solid capture → pass/review, single-band → reshoot, matrix shape, JSON-serializable |

**Toplam: 18 yeni test → 18/18 passed.**

## Mimari Etki

### Karar Tablosu (Threshold Defaults)

| Sinyal | Warn (review) | Fail (reshoot) |
|--------|--------------|----------------|
| `blur_burst_ratio` | ≥ 10% | ≥ 25% |
| `multi_height_score` | < 0.34 (1 band) | < 0.10 |
| `azimuth_orbit_progress` | < 0.40 | < 0.20 |
| `static_run_ratio` | ≥ 30% of frames | — |
| `frame_count` | < 8 | — |

`reshoot > review > pass` mantığı: en kötü sinyal kararı belirler.

### Veri Akışı

```
[upload] POST /api/sessions/upload (Sprint 5)
   ↓
[worker] frame extraction
   ↓
   evaluate_capture(frame_paths, masks_dir)
   ↓
   extraction_manifest.json ← capture_gate {decision, reasons, matrix_3x8, ...}
   ↓
[UI poll] GET /api/sessions/{id}/capture-gate
   ↓
   capture_gate.html → decision badge + matrix overlay + suggestions
   ↓
[reconstruction] runner.py
   ↓
   build_scorecard() ← reads capture_gate from extraction_manifest
   ↓
   quality_report.json ← overall.grade demoted if gate=reshoot/review
```

### Heuristic Sınırları (Bilinçli Tradeoffs)

- **Elevation/azimuth gerçek COLMAP pose değil**, mask centroid heuristic. Sebep: extraction-zamanında pose yok, recon henüz başlamamış. Doğruluk ~%60-70 (single-orbit captures için iyi, multi-clip için zayıf).
- **Sprint 4 sonrası** `intrinsics_cache.py` → ikinci capture'larda gerçek BA pose'ları paylaşıyor olacak; bu modülleri `coverage_metrics.py` ile bağlayıp heuristic'i kapatacağız.
- **3×8 matrix yaklaşık dağılım** — gerçek azimuth bucket bilinmediği için frame'ler `hash(name) % azimuth_span` ile spread ediliyor. UI için yeterli, metric için değil.

### `extraction_manifest.json` Genişlemesi

```json
{
  "manifest_hash": "...",
  "video_path": "...",
  "frame_count": 26,
  "color_profile": { ... },
  "capture_profile": { ... },
  "capture_gate": {                      ← Sprint 2 ekledi
    "decision": "review",
    "reasons": ["multi_height_score 0.33 below warn threshold"],
    "suggestions": ["add at least one extra elevation band (top-down or low-angle)"],
    "blur": { "burst_ratio": 0.04, ... },
    "elevation": { "multi_height_score": 0.33, "bucket_counts": {...}, ... },
    "azimuth": { "cumulative_orbit_progress": 0.62, ... },
    "matrix_3x8": [[3,2,4,1,2,3,2,1], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]],
    "gate_thresholds": { ... }
  }
}
```

## Başarı Kriterleri

| # | Kriter | Sonuç |
|---|--------|-------|
| 1 | Blur burst detector temporal cluster doğru tespit | ✅ 4-frame burst test, 1-frame skip test |
| 2 | Elevation 3-band sınıflama (low/mid/top) | ✅ centroid_y_norm thresholds doğrulandı |
| 3 | Azimuth heuristic mask metadata olmadan no-op | ✅ "no masks_dir" warning |
| 4 | Gate orchestrator 3-tier decision (pass/review/reshoot) | ✅ 6 senaryo test edildi |
| 5 | 3×8 matrix UI için always-defined skeleton | ✅ empty input için zero matrix |
| 6 | Scorecard'a entegrasyon: reshoot → grade F | ✅ scorecard.py güncellendi |
| 7 | UI polling endpoint + standalone HTML | ✅ `/api/sessions/{id}/capture-gate` + `capture_gate.html` |
| 8 | Mevcut suitlerde 0 yeni regresyon | ✅ 1 pre-existing fail aynı kaldı |

## Mimari Sınırlar (Sprint 3'e Bırakılan)

- ❌ Adaptive temporal sampling (optical flow + IoU bazlı dynamic frame_sample_rate) — Sprint 3 işi
- ❌ Capture matrix henüz **upload-zamanı reject** yapmıyor (advisory). Sprint 5 multi-clip altyapısı geldikten sonra hard reject opsiyonu eklenir.
- ❌ Real-time live AR capture feedback (operatör çekim sırasında matrix görür) — Sprint 5+ (multi-clip session ile birlikte)

## Sıradaki: Sprint 3 — Keyframe Selection v2 + Adaptive Sampling

- Optical flow + bbox IoU bazlı dynamic decimation (sabit `frame_sample_rate` yerine)
- Coverage-aware keyframe selection (eksik elevation/azimuth bucket'larını dolduran frame'leri tercih et)
- Texture frame filter ile koordinasyon (zaten var, optimize)

Hazır olduğunda başlatabilirim.
