# RUNBOOK: Meshysiz Product Asset Factory

Bu doküman, sistemin günlük operasyonlarını ve olası hata durumlarını (troubleshooting) yönetmenizi sağlar.

## 0. Kurulum ve Ortam Yönetimi (Setup)

Sistemin bağımlılıklarını yönetmek ve modül yollarını doğru tanımak için aşağıdaki adımları izleyin. Her zaman projenin kök dizininde olduğunuzdan emin olun.

### 0.1 Bağımlılıkların Kurulumu
Bağımlılıkları kurarken spesifik Python interpreter'ını kullanın:
```powershell
C:\Users\Ömer\anaconda3\python.exe -m pip install rembg onnxruntime
```

### 0.2 Projeyi "Editable" Modda Kurma
Modül bazlı çağırmaların (`import modules...`) sorunsuz çalışması için projeyi geliştirme modunda kurun:
```powershell
C:\Users\Ömer\anaconda3\python.exe -m pip install -e .
```

> [!NOTE]
> `npm install` komutu şu an için gerekli değildir. UI statik dosyalardan oluşmaktadır. `package.json` eklenene kadar bu adımı atlayabilirsiniz.

## 1. Operasyonel Akış

### 1.1 Çalıştırma ve Modül Bazlı Çağrılar
Script'leri doğrudan dosya yoluyla değil, `-m` bayrağı ile bir modül olarak çalıştırın. Bu, `ModuleNotFoundError` hatalarını engeller.

**Dashboard Başlatma:**
```powershell
python dashboard.py
```

**Bileşenleri İzole Test Etme (Örnek):**
```powershell
python -m modules.capture_workflow.object_masker
```

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

### `Recapture Required` (Status)
- **Sebep**: Maskeler bulunamadı (`missing_mask`), maske kalitesi düşük (`bad_mask`) veya viewpoint çeşitliliği yetersiz.
- **Detay**: `COLMAPAdapter` veya `CoverageAnalyzer` aşamalarında tetiklenebilir. Loglarda `modes(stem=X, legacy=Y, none=Z)` şeklinde maske çözümleme detayları görülebilir.
- **Aksiyon**: Capture kalitesini artırın veya objenin kameraya her yönden (360 derece) göründüğünden emin olun.

---

## 3. İzleme (Monitoring)
- **Log Konumu**: Tüm işlemler `data/reconstructions/{job_id}/logs/` altında kaydedilir.
- **Audit Log**: `AssetRegistry` içindeki `audit_logs` her asset'in tüm yaşam döngüsünü (kayıt, onay, paketlenme) takip eder.

---

## 4. İletişim Matrixi
- **Sistem Hatası**: Teknik Destek Ekibi.
- **Kalite Sorunu**: İçerik / QA Ekibi.
