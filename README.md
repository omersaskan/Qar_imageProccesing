# Meshysiz Product Asset Factory

**Meshysiz Product Asset Factory**, restoran ürünleri için yüksek kaliteli, AR-uyumlu 3D modeller üreten modüler bir boru hattıdır (pipeline). Bu sistem, kısa videolardan (guided short video) başlayarak temizlenmiş, optimize edilmiş ve doğrulanmış dijital varlıklar üretir.

---

## 1. Temel Özellikler
- **Guided Capture**: Keyframe çıkarımı ve coverage analizi ile rehberli çekim.
- **Automated Reconstruction**: Ham 3D verinin (mesh & texture) otomatik üretimi.
- **Cleanup Pipeline**: Mesh temizleme, remesh, pivot düzeltme ve BBox çıkarımı.
- **Multi-Format Export**: GLB, USDZ, Poster (PNG) ve Thumbnail üretim.
- **Quality Gates**: Polycount, alignment ve texture kontrolü ile otomatik/manuel onay mekanizması.
- **Asset Registry**: Versiyonlama, rollback ve "Source of Truth" metadata yönetimi.

---

## 2. Hızlı Başlangıç

### Gereksinimler
- Python 3.11+
- Pip

### Kurulum
```bash
# Bağımlılıkları yükle
pip install -r requirements.txt
# Geliştirme/Test bağımlılıkları için
pip install -e .[dev]
```

---

## 3. Kullanım (Canonical Commands)

### Test Suite Çalıştırma
Projenin stabilitesini doğrulamak için tüm testleri çalıştırın:
```bash
python -m pytest modules/shared_contracts/tests modules/capture_workflow/tests modules/reconstruction_engine/tests modules/asset_cleanup_pipeline/tests modules/export_pipeline/tests modules/qa_validation/tests modules/asset_registry/tests modules/tests/test_phase6_integration.py modules/tests/test_smoke_flow.py modules/tests/test_phase7_edge_cases.py
```

### Smoke Test (E2E Flow)
Uçtan uca capture -> publish akışını doğrulamak için:
```bash
python -m pytest modules/tests/test_smoke_flow.py
```

---

## 4. Proje Yapısı
```text
project/
├── modules/                # Ana Modüller
│   ├── shared_contracts/   # Veri modelleri ve hata tipleri
│   ├── capture_workflow/   # Video işleme ve keyframe setleri
│   ├── reconstruction/     # 3D üretimi orkestrasyonu
│   ├── asset_cleanup/      # Mesh temizleme ve normalizasyon
│   ├── export_pipeline/    # Dosya formatları ve statik medya
│   ├── qa_validation/      # Kalite kontrol ve validation rules
│   └── asset_registry/     # Varlık kütüphanesi ve yayıncılık
├── tools/                  # Yardımcı debug ve analiz araçları
├── tests/                  # Entegrasyon ve duman testleri
└── data/                   # Yerel işleme dizini (ignore edilmiş)
```

---

## 5. Dokümantasyon
- [ARCHITECTURE.md](ARCHITECTURE.MD): Sistemin modüler mimarisi ve veri akışı.
- [HANDOFF.md](HANDOFF.md): Teknik devir notları ve bağımlılık detayları.
- [RUNBOOK.md](RUNBOOK.md): Operasyonel rehber ve hata giderme.

---

## 6. Lisans ve Katkıda Bulunma
Bu proje **Meshysiz Team** tarafından geliştirilmiştir. Harici paylaşımlar ve katkılar için teknik lider ile iletişime geçiniz.
