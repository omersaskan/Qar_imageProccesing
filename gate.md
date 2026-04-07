# gate.md
# Agent Gate — Meshysiz Product Asset Factory (STATUS: FINALIZED)

## 1. Global Kural
Bu proje bir **AR runtime projesi değil**, bir **3D asset factory** projesidir.

Agent yalnızca şu alanlarda çalışmalıdır:
- capture
- keyframe extraction
- reconstruction
- cleanup
- glb/usdz export
- poster / thumbnail üretimi
- metadata
- validation
- registry
- package publish

AR runtime, calibration, no-LiDAR state machine ve uygulama fallback logic bu projenin scope'u dışındadır.

---

## 2. İlk Kontrol Soruları
1. Bu görev asset factory scope'unda mı?
2. Bu görev hangi modüle ait?
3. Aynı logic başka modülde zaten var mı?
4. Raw output ile final delivery output ayrımı korunuyor mu?
5. Registry source of truth korunuyor mu?
6. Validation atlanıyor mu?
7. Version bütünlüğü bozuluyor mu?

---

## 3. Gate'ler

### GATE-01 — Scope Boundary Gate
#### Geçme Koşulu
Görev yalnızca aşağıdakilerden biriyle ilgili olmalı:
- capture
- keyframe extraction
- reconstruction
- cleanup
- export
- poster/thumbnail
- registry
- qa/validation
- publish

#### Fail
- AR runtime'a giriyorsa
- device logic ekliyorsa
- app içi fallback / UI yapıyorsa

---

### GATE-02 — Module Ownership Gate
#### Geçme Koşulu
- capture-workflow yalnızca capture ve keyframe ile ilgilenir
- reconstruction-engine yalnızca ham 3D üretir
- asset-cleanup-pipeline yalnızca cleanup/optimization yapar
- export-pipeline yalnızca GLB/USDZ/poster/thumbnail üretir
- asset-registry yalnızca kayıt / sürüm / publish yönetir
- qa-validation yalnızca kalite kararı verir

#### Fail
- Aynı logic iki modülde tekrar ediyorsa
- Modül sınırı ihlal ediliyorsa

---

### GATE-03 — Raw vs Delivery Separation Gate
#### Geçme Koşulu
- Raw reconstruction output ile final delivery output ayrı tutulur
- Cleanup ve validation olmadan publish yapılmaz

#### Fail
- Raw mesh doğrudan final sayılıyorsa

---

### GATE-04 — Registry Source of Truth Gate
#### Geçme Koşulu
- Active asset registry'den geliyor
- Export URL'leri registry'de tutuluyor
- Physical profile registry ile bağlı
- Publish state registry'de

#### Fail
- Hardcoded URL
- Registry dışı gizli veri kaynağı
- Direkt pipeline çıktısından kullanım

---

### GATE-05 — Version Integrity Gate
#### Geçme Koşulu
Aynı asset version için birlikte bağlı olanlar:
- GLB
- USDZ
- poster
- thumbnail
- bbox
- pivot
- physical profile
- validation status

#### Fail
- Bu çıktılar farklı sürümlerden geliyorsa

---

### GATE-06 — Contract Integrity Gate
#### Geçme Koşulu
- Yeni veri alanları shared-contracts üzerinden tanımlanır
- Modüller ad hoc veri formatı üretmez

#### Fail
- Farklı modüllerde farklı model tanımları varsa

---

### GATE-07 — Capture Quality Gate
#### Geçme Koşulu
- Guided short video akışı korunur
- Kalite scoring vardır
- Keyframe extraction quality-aware çalışır
- Coverage kontrolü vardır

#### Fail
- Kötü capture reconstruction'a körlemesine gidiyorsa

---

### GATE-08 — Cleanup Gate
#### Geçme Koşulu
- Pivot correction var
- Ground alignment var
- BBox extraction var
- Optimization var

#### Fail
- Cleanup atlanmışsa
- Mobil dostu olmayan model final kabul ediliyorsa

---

### GATE-09 — Export Completeness Gate
#### Geçme Koşulu
- GLB var
- USDZ var
- poster var
- thumbnail var
- bunlar package içinde bağlı

#### Fail
- Eksik paket publish ediliyorsa

---

### GATE-10 — Validation Gate
#### Geçme Koşulu
- Validation report var
- pass / fail / review ayrımı var
- QA yolu tanımlı

#### Fail
- Validation olmadan publish
- Review gereken durumda otomatik onay

---

### GATE-11 — Observability Gate
#### Geçme Koşulu
- Capture fail reason loglanıyor
- Reconstruction fail reason loglanıyor
- Validation sonucu loglanıyor
- Export fail reason loglanıyor
- Publish state loglanıyor

#### Fail
- Hata olduğunda neden görünmüyorsa

---

### GATE-12 — Testability Gate
#### Geçme Koşulu
- Modüller testlenebilir interface sunar
- Keyframe extraction testlenebilir
- Registry publish koşulları doğrulanabilir
- Validation kararları testlenebilir

#### Fail
- Kritik kurallar test edilemiyorsa

---

## 4. Done Definition
Bir görev ancak şunlar sağlanırsa tamam sayılır:
- doğru modülde uygulanmış
- scope dışına taşmamış
- contract ihlali yok
- registry bypass edilmemiş
- validation etkisi düşünülmüş
- version bütünlüğü korunmuş
- loglama etkisi düşünülmüş
- test yaklaşımı not edilmiş

---

## 5. Yasaklar
Agent şunları yapmamalıdır:
- AR runtime kapsamına girmek
- raw mesh'i final asset diye işaretlemek
- validation olmadan publish yapmak
- registry'yi bypass etmek
- farklı sürümlerden karışık package oluşturmak
- hardcoded export URL kullanmak
- shared-contracts dışı veri modeli uydurmak
