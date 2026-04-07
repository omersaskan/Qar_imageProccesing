# HANDOFF: Meshysiz Product Asset Factory

Bu doküman, sistemin teknik devrini ve modüler yapısını detaylandırır.

## 1. Mimari Prensipler
- **Modülerlik**: Modüller sadece `shared_contracts` üzerinden veri dili konuşur.
- **Source of Truth**: `AssetRegistry`, asset sürümleri ve aktif yayın durumları için tek yetkili kaynaktır.
- **Validation Gates**: Kalite raporu `pass` olmayan hiçbir varlık, manuel onay olmadan yayınlanamaz. `fail` alanlar terminal olarak bloklanır.

## 2. Modüler Bağımlılık Grafiği
```text
capture_workflow -> reconstruction_engine -> cleanup_pipeline
    |                     |                         |
    v                     v                         v
shared_contracts <--- export_pipeline <--- qa_validation
                           |
                           v
                     asset_registry
```

## 3. Veri Akışı (Lifecycle)
1. **Created**: `CaptureSession` başlatıldı.
2. **Captured**: Videodan keyframe setleri çıkarıldı.
3. **Reconstructed**: Ham 3D mesh ve texture üretildi.
4. **Cleaned**: Mesh temizlendi, pivot düzeltildi ve metadata oluştu.
5. **Exported**: GLB, USDZ ve medya (PNG) sürümleri hazırlandı.
6. **Validated**: Otomatik kurallar işletildi (PASS/FAIL/REVIEW).
7. **Published**: Registry'ye kaydedildi ve aktif sürüm yapıldı.

## 4. Teknik Notlar
- **Pydantic Hardening**: Tüm veri modelleri pozitif boyut ve mantıklı polycount değerleri için zorunlu kılınmıştır.
- **Path Safety**: Girdi ve çıktı dizinleri için `../` traversal koruması (`validate_safe_path`) her noktada entegredir.
- **Windows Uyumluluğu**: Tüm dosya yolu işlemleri `os.path.join` ve mutlak yol çözünürlüğü ile yapılmıştır.

## 5. İletişim ve Destek
Sistemin mimarisi ve çekirdek (core) modülleri için **Mimar / Teknik Lider** ile iletişime geçiniz.
