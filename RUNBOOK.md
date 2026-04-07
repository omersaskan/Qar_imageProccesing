# RUNBOOK: Meshysiz Product Asset Factory

Bu doküman, sistemin günlük operasyonlarını ve olası hata durumlarını (troubleshooting) yönetmenizi sağlar.

## 1. Operasyonel Akış

### 1.1 Model Onaylama
`ValidationReport` statüsü `review` olan asset'ler manuel onay bekler.
```python
from modules.asset_registry.registry import AssetRegistry
registry = AssetRegistry()
# asset_id onay ver
registry.grant_approval("asset_id_v1", "review")
```

### 1.2 Model Geri Çekme (Rollback)
Hatalı bir sürüm kazara yayınlanırsa son stabil sürüme dönmek için:
```python
registry.rollback_version("product_id")
```

---

## 2. Hata Giderme (Troubleshooting)

### `ERR_VALIDATION_FAIL` (Hata Kodu)
- **Sebep**: Polycount sınırın üstünde, model yere oturmuyor veya texture eksik.
- **Aksiyon**: `qa_validation/rules.py` içindeki eşikleri kontrol edin veya asset'i "rework" olarak işaretleyip yeni capture alın.

### `MetadataCorruptionError`
- **Sebep**: `normalized_metadata.json` dosyası eksik veya bozulmuş.
- **Aksiyon**: Cleanup pipeline'ı manuel olarak tekrar tetikleyin. Otomatik onarım (auto-repair) güvenlik nedeniyle kapalıdır.

### `DuplicateAssetError`
- **Sebep**: Aynı `asset_id` ile ikinci bir kayıt denemesi.
- **Aksiyon**: `AssetRegistry` içindeki mevcut sürümleri kontrol edin. Her sürümün benzersiz bir ID'ye sahip olması zorunludur.

---

## 3. İzleme (Monitoring)
- **Log Konumu**: Tüm işlemler `data/reconstructions/{job_id}/logs/` altında kaydedilir.
- **Audit Log**: `AssetRegistry` içindeki `audit_logs` her asset'in tüm yaşam döngüsünü (kayıt, onay, paketlenme) takip eder.

---

## 4. İletişim Matrixi
- **Sistem Hatası**: Teknik Destek Ekibi.
- **Kalite Sorunu**: İçerik / QA Ekibi.
