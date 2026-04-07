# tasks.md
# Meshysiz Product Asset Factory — Görev Listesi

Bu dosya, agent'ın projeyi modüler ve kontrollü şekilde inşa etmesi için görevleri sıralar.

## Görev Yazım Kuralları
- Her görev tek bir modül odaklı olmalı
- Görev tamamlanmadan önce gate.md kontrol edilmeli
- Raw output ile final delivery output ayrımı korunmalı
- Contract değişikliği gerekiyorsa önce shared-contracts güncellenmeli

---

## Faz 1 — Temel Omurga

### T-001 — Proje iskeletini kur
**Amaç:** Temel klasör yapısını oluşturmak.

**Çıktı**
- `modules/` altında başlangıç modülleri
- ortak lint/test config
- environment ayırımı

**Bağımlılık:** yok

---

### T-002 — shared-contracts modülünü oluştur
**Amaç:** Ortak veri modellerini ve şemaları tanımlamak.

**Tanımlanacak yapılar**
- Product
- CaptureSession
- ReconstructionJob
- AssetMetadata
- ProductPhysicalProfile
- ValidationReport
- AssetPackage

**Bağımlılık:** T-001

---

### T-003 — Asset lifecycle enum ve durum makinesini tanımla
**Amaç:** Asset durumlarını standardize etmek.

**Durumlar**
- created
- captured
- reconstructed
- cleaned
- exported
- validated
- published

**Bağımlılık:** T-002

---

## Faz 2 — Capture Tarafı

### T-004 — capture-workflow modülünü oluştur
**Amaç:** Capture session açma / kapama altyapısını kurmak.

**Yapılacaklar**
- session manager
- capture session kayıt modeli
- ürün bazlı session başlatma akışı

**Bağımlılık:** T-002

---

### T-005 — Guided short video akışını oluştur
**Amaç:** Operatörün kısa video ile capture yapmasını sağlamak.

**Yapılacaklar**
- rehberli capture ekran mantığı
- minimum çekim yönergeleri
- capture session ile video bağlama

**Bağımlılık:** T-004

---

### T-006 — Capture kalite ölçüm katmanını ekle
**Amaç:** Kötü videoyu erken aşamada elemek.

**Yapılacaklar**
- blur score
- exposure score
- framing check
- occlusion check
- accept / retry sonucu

**Bağımlılık:** T-005

---

### T-007 — Otomatik keyframe extraction modülünü yaz
**Amaç:** Videodan reconstruction için kullanılacak frame setini üretmek.

**Yapılacaklar**
- frame sampling
- similarity filter
- blur filter
- quality-aware selection

**Bağımlılık:** T-006

---

### T-008 — Coverage analyzer yaz
**Amaç:** Yetersiz açıyla reconstruction yapılmasını önlemek.

**Yapılacaklar**
- açı dağılımı kontrolü
- üst açı kontrolü
- tekrar capture önerisi

**Bağımlılık:** T-007

---

### T-009 — Reconstruction input packager oluştur
**Amaç:** Keyframe setini reconstruction-engine için standart payload'a çevirmek.

**Bağımlılık:** T-007, T-008

---

## Faz 3 — Reconstruction Tarafı

### T-010 — reconstruction-engine modülünü oluştur
**Amaç:** Reconstruction job lifecycle altyapısını kurmak.

**Yapılacaklar**
- job queue interface
- job state management
- failure reason standardı

**Bağımlılık:** T-002, T-009

---

### T-011 — Input preprocess pipeline yaz
**Amaç:** Keyframe setini reconstruction öncesi normalize etmek.

**Yapılacaklar**
- resize
- denoise
- renk normalize
- opsiyonel background hazırlığı

**Bağımlılık:** T-010

---

### T-012 — Geometric reconstruction akışını bağla
**Amaç:** Keyframe setinden ham 3D sonuç üretmek.

**Yapılacaklar**
- reconstruction entrypoint
- raw mesh üretimi
- raw texture üretimi
- log/artefact kayıtları

**Bağımlılık:** T-011

---

### T-013 — Reconstruction hata yönetimini ekle
**Amaç:** Teknik ve veri kaynaklı hataları kontrollü hale getirmek.

**Yapılacaklar**
- failure reason mapping
- retry önerisi
- operatöre dönülecek minimum bilgi

**Bağımlılık:** T-012

---

## Faz 4 — Cleanup ve Export

### T-014 — asset-cleanup-pipeline modülünü oluştur
**Amaç:** Raw modelin cleanup ve optimize akışını kurmak.

**Bağımlılık:** T-012

---

### T-015 — Mesh cleaning / remesh / decimation uygula
**Amaç:** Raw mesh'i mobil dostu hale getirmek.

**Bağımlılık:** T-014

---

### T-016 — Pivot correction ve ground alignment uygula
**Amaç:** Modelin doğru oturum ve orijine sahip olmasını sağlamak.

**Bağımlılık:** T-015

---

### T-017 — BBox ve metadata extraction ekle
**Amaç:** Asset metadata üretmek.

**Bağımlılık:** T-016

---

### T-018 — export-pipeline modülünü oluştur
**Amaç:** Final modelden teslim formatları üretmek.

**Bağımlılık:** T-017

---

### T-019 — GLB export üret
**Amaç:** Android / genel 3D kullanım için GLB çıktısı vermek.

**Bağımlılık:** T-018

---

### T-020 — USDZ export üret
**Amaç:** iOS tarafı için USDZ çıktısı vermek.

**Bağımlılık:** T-018

---

### T-021 — Poster ve thumbnail üret
**Amaç:** Statik görsel çıktıları oluşturmak.

**Bağımlılık:** T-018

---

## Faz 5 — Validation ve Registry

### T-022 — qa-validation modülünü oluştur
**Amaç:** Asset kalite kontrol katmanını kurmak.

**Bağımlılık:** T-017, T-018

---

### T-023 — Validation kurallarını yaz
**Amaç:** Pass / fail / review karar mekanizmasını oluşturmak.

**Kurallar**
- polycount
- texture eksikliği
- bbox mantığı
- ground alignment
- dosya açılabilirliği

**Bağımlılık:** T-022

---

### T-024 — asset-registry modülünü oluştur
**Amaç:** Asset, metadata ve sürüm bilgisini kayıt altına almak.

**Bağımlılık:** T-002, T-018, T-022

---

### T-025 — Active version ve rollback mantığını yaz
**Amaç:** Ürün bazlı aktif asset sürümü yönetmek.

**Bağımlılık:** T-024

---

### T-026 — Physical profile yönetimini ekle
**Amaç:** Gerçek dünya ölçülerini asset ile ilişkilendirmek.

**Bağımlılık:** T-024

---

### T-027 — Ready-for-AR package publisher yaz
**Amaç:** Nihai teslim paketini üretmek.

**Bağımlılık:** T-019, T-020, T-021, T-023, T-024, T-026

---

## Faz 6 — Yönetim ve Operasyon

### T-028 — QA review akışını ekle
**Amaç:** İnsan onaylı karar sürecini desteklemek.

**Bağımlılık:** T-023, T-024

---

### T-029 — Asset sürümleme ekran / servis mantığını ekle
**Amaç:** Sürüm geçmişi ve rollback görünürlüğü sağlamak.

**Bağımlılık:** T-025

---

### T-030 — Operasyonel log ve telemetry ekle
**Amaç:** Hata ve kalite takibini mümkün kılmak.

**Bağımlılık:** T-006, T-013, T-023, T-027

---

## Faz 7 — Hardening

### T-031 — Modül testlerini ekle
**Amaç:** Kritik kuralları test altına almak.

**Test edilecekler**
- keyframe extraction
- coverage analyzer
- asset lifecycle
- validation kararları
- package publish koşulları

**Bağımlılık:** tüm temel modüller

---

### T-032 — End-to-end teslim akışını doğrula
**Amaç:** Capture'dan ready-for-AR package'e kadar tam zincirin çalıştığını kanıtlamak.

**Bağımlılık:** T-031

---

## Son Not
Agent görevleri sırayla ele almalı; reconstruction ve export oturmadan registry/publish katmanına geçmemeli.
