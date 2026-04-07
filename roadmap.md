# roadmap.md
# Meshysiz Product Asset Factory — Yol Haritası

Bu dosya, projenin hangi sırayla ve hangi olgunluk seviyesinde inşa edilmesi gerektiğini tanımlar.

## 1. Hedef
Amaç, ürünlerden **GLB + USDZ + poster/thumbnail + metadata** üreten ve bunları **ready-for-AR asset package** olarak teslim eden bir sistem kurmaktır.

---

## 2. Yol Haritası Özeti

### Faz 1 — Temel Veri ve Modül Sınırları [TAMAMLANDI]
Hedef:
- veri modellerini netleştirmek
- modül sahipliklerini belirlemek
- asset lifecycle'ı standardize etmek

Teslim:
- shared-contracts
- temel klasör yapısı
- lifecycle enum'ları

Başarı Kriteri:
- tüm modüller aynı veri dilini kullanıyor
- **Durum:** ✅ Doğrulandı.































---

### Faz 2 — Capture ve Input Kalitesi [TAMAMLANDI]
Hedef:
- guided short video sürecini kurmak
- kötü input'u erken elemek
- reconstruction için temiz keyframe seti üretmek

Teslim:
- capture-workflow
- kalite skorlama
- keyframe extraction
- coverage analyzer

Başarı Kriteri:
- operatör kısa videodan reconstruction-ready frame seti çıkarabiliyor
- **Durum:** ✅ Doğrulandı.

---

### Faz 3 — Reconstruction Çekirdeği [TAMAMLANDI]
Hedef:
- çoklu kareden ham 3D çıktı üretmek
- job lifecycle kurmak
- failure reason'ları görünür hale getirmek

Teslim:
- reconstruction-engine
- preprocess
- raw mesh/raw texture çıktıları
- retry/fail mantığı

Başarı Kriteri:
- sistem en azından raw 3D çıktı üretebiliyor
- **Durum:** ✅ Doğrulandı.

---

### Faz 4 — Cleanup ve Export [TAMAMLANDI]
Hedef:
- raw çıktıyı teslim edilebilir hale getirmek
- GLB/USDZ ve statik medya üretmek

Teslim:
- asset-cleanup-pipeline
- export-pipeline
- GLB export
- USDZ export
- poster/thumbnail üretimi

Başarı Kriteri:
- raw model yerine optimize edilmiş teslim paketi üretilebiliyor
- **Durum:** ✅ Doğrulandı.

---

### Faz 5 — Validation ve Registry [TAMAMLANDI]
Hedef:
- kalite kapısı koymak
- asset versiyonlarını yönetmek
- nihai package oluşturmak

Teslim:
- qa-validation
- asset-registry
- active version
- rollback
- ready-for-AR package publisher

Başarı Kriteri:
- bir ürün için tek bir güvenilir published asset package var
- **Durum:** ✅ Doğrulandı.

---

### Faz 6 — Operasyonel Olgunluk [TAMAMLANDI]
Hedef:
- QA akışını oturtmak
- log/telemetry toplamak
- ekip kullanımını kolaylaştırmak

Teslim:
- QA review süreci
- operasyonel loglar
- asset geçmişi görünürlüğü

Başarı Kriteri:
- ekip üretim hattısını düzenli kullanabiliyor
- **Durum:** ✅ Doğrulandı.

---

### Faz 7 — Stabilizasyon [TAMAMLANDI]
Hedef:
- modül testleri
- uçtan uca akış doğrulaması
- publish öncesi güven seviyesini artırmak

Teslim:
- modül testleri
- end-to-end doğrulama
- publish gate sağlamlaştırma

Başarı Kriteri:
- capture'dan delivery package'e kadar akış tekrar edilebilir ve güvenilir
- **Durum:** ✅ Doğrulandı (45/45 Passed).

---

## 3. Öncelik Sırası

### Önce yapılmalı
1. shared-contracts
2. capture-workflow
3. reconstruction-engine

### Sonra yapılmalı
4. asset-cleanup-pipeline
5. export-pipeline

### Son aşamada
6. qa-validation
7. asset-registry
8. package publish

### En son
9. QA review
10. telemetry
11. hardening

---

## 4. MVP Tanımı
MVP için gereken minimum parçalar:

- shared-contracts
- capture-workflow
- reconstruction-engine
- asset-cleanup-pipeline
- export-pipeline
- qa-validation
- asset-registry
- ready-for-AR package publish

### MVP Çıktısı
Bir ürün için:
- GLB
- USDZ
- poster
- thumbnail
- bbox/pivot/physical profile
- validation sonucu
- ready-for-AR package

---

## 5. V1 Sonrası Gelişim
MVP sonrası iyileştirmeler:
- capture önerilerini daha akıllı hale getirme
- reconstruction kalite skorunu artırma
- cleanup otomasyonunu geliştirme
- export optimizasyon profilleri
- QA araçlarını geliştirme
- asset kalite dashboard'u

---

## 6. Yaygın Hatalar
Bu yol haritasında özellikle kaçınılması gerekenler:

- scope'u AR tarafına kaydırmak
- raw çıktıyı final diye kullanmak
- validation olmadan publish yapmak
- registry'yi bypass etmek
- farklı sürümlerden paket oluşturmak
- keyframe extraction'ı basit frame sampling'e indirmek

---

## 7. Nihai Başarı Tanımı
Proje başarılı sayılırsa:
- operatör kısa videodan içerik üretebiliyor
- sistem reconstruction ve cleanup yapabiliyor
- GLB ve USDZ güvenilir şekilde çıkıyor
- poster/thumbnail üretiliyor
- validation sonrası tekil asset package publish ediliyor
- AR ekibi bu paketi ek iş yapmadan tüketebiliyor
