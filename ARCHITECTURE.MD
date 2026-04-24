# ARCHITECTURE.MD [FINALIZED]
# Meshysiz Product Asset Factory Mimarisi

## 1. Mimari Sınır
Bu proje yalnızca **3D asset üretim hattısıdır**.

### Dahil
- Capture
- Guided short video
- Keyframe extraction
- Reconstruction
- Cleanup / optimization
- GLB export
- USDZ export
- Poster / thumbnail üretimi
- Asset metadata
- Physical profile
- Validation
- Asset versioning
- Ready-for-AR package publish

### Dahil Değil
- AR runtime
- calibration
- placement
- no-LiDAR davranışı
- video fallback
- uygulama içi hata / retry UI

---

## 2. Temel Prensipler
1. **Asset factory only**
2. **Raw output ile final delivery output kesin ayrılmalı**
3. **Registry source of truth olmalı**
4. **GLB/USDZ ve statik medya aynı asset version'a bağlı olmalı**
5. **Human-in-the-loop desteklenmeli**
6. **Quality gate olmadan publish yok**

---

## 3. Modül Yapısı

```text
project/
  modules/
    shared-contracts/
    capture-workflow/
    reconstruction-engine/
    asset-cleanup-pipeline/
    export-pipeline/
    asset-registry/
    qa-validation/
```

---

## 4. Modüller

### 4.1 shared-contracts
#### Görev
Tüm modüllerin ortak veri dilini tanımlamak.

#### Yapılar
- Product
- CaptureSession
- ReconstructionJob
- AssetMetadata
- ProductPhysicalProfile
- ValidationReport
- AssetPackage

#### Kural
- Başka modüllere bağımlı olmamalı

---

### 4.2 capture-workflow
#### Görev
Guided short video tabanlı capture sürecini yönetmek.

#### Sorumluluklar
- Capture session başlatma
- Rehberli video akışı
- Canlı kalite uyarıları
- Keyframe extraction
- Coverage analizi
- Reconstruction input hazırlama

#### Çıktı
- capture session
- keyframe seti
- reconstruction payload

---

### 4.3 reconstruction-engine
#### Görev
Keyframe setinden ham 3D çıktı üretmek.

#### Sorumluluklar
- Job orchestration
- Preprocess
- Geometric reconstruction
- Meshing
- Texturing
- Raw export
- Failure logging

#### Çıktı
- raw mesh
- raw texture
- reconstruction logs

#### Kural
- Production-ready output vermez

---

### 4.4 asset-cleanup-pipeline
#### Görev
Raw 3D çıktıyı teslim edilebilir hale getirmek.

#### Sorumluluklar
- Mesh cleaning
- Remesh / decimation
- Pivot correction
- Ground alignment
- BBox extraction
- Texture optimization

#### Çıktı
- optimized mesh
- bbox
- pivot
- normalized metadata

---

### 4.5 export-pipeline
#### Görev
Final mesh’i teslim formatlarına çevirmek.

#### Sorumluluklar
- GLB export
- USDZ export
- Poster render
- Thumbnail üretimi
- Storage upload

#### Çıktı
- glb url
- usdz url
- posterImageUrl
- thumbnailUrl

---

### 4.6 asset-registry
#### Görev
Asset ve metadata için resmi doğruluk kaynağı olmak.

#### Sorumluluklar
- ürün → asset eşleşmesi
- active asset version
- rollback
- physical profile bağlama
- package publish state

#### Kural
- Registry olmadan publish yok

---

### 4.7 qa-validation
#### Görev
Kötü asset’in teslim paketine girmesini engellemek.

#### Sorumluluklar
- polycount kontrolü
- texture kontrolü
- bbox kontrolü
- ground alignment kontrolü
- dosya açılabilirlik testi
- pass / fail / review kararı

---

## 5. Veri Akışı

```text
capture-workflow
  -> reconstruction-engine
  -> asset-cleanup-pipeline
  -> export-pipeline
  -> qa-validation
  -> asset-registry
  -> ready-for-ar asset package
```

---

## 6. Ana Veri Modelleri

### CaptureSession
- sessionId
- productId
- operatorId
- sourceVideoUrl
- extractedFrames
- coverageScore
- status

### ReconstructionJob
- jobId
- captureSessionId
- status
- rawMeshUrl
- rawTextureUrl
- failureReason

### AssetMetadata
- assetId
- productId
- version
- bbox
- pivotOffset
- qualityGrade

### ProductPhysicalProfile
- realWidthCm
- realDepthCm
- realHeightCm
- plateDiameterCm
- groundOffsetCm
- recommendedScaleMultiplier

### ValidationReport
- assetId
- polyCount
- textureStatus
- bboxReasonable
- groundAligned
- mobilePerformanceGrade
- finalDecision

### AssetPackage
- productId
- assetVersion
- glbUrl
- usdzUrl
- posterImageUrl
- thumbnailUrl
- bbox
- pivotOffset
- physicalProfile
- validationStatus
- packageStatus

---

## 7. Asset Lifecycle
1. Created
2. Captured
3. Reconstructed
4. Cleaned
5. Exported
6. Validated
7. Published

---

## 8. Mimari Kurallar
1. Capture modülü export yapmaz
2. Reconstruction modülü aktif asset seçmez
3. Cleanup modülü admin kararı vermez
4. Export modülü QA kararı vermez
5. Validation pass olmadan package `ready_for_ar` olamaz
6. Poster / thumbnail ile GLB/USDZ aynı version'a bağlı olmalıdır

---

## 9. MVP
### Faz 1
- shared-contracts
- capture-workflow
- reconstruction-engine v1

### Faz 2
- asset-cleanup-pipeline
- export-pipeline

### Faz 3
- qa-validation
- asset-registry
- package publish

---

## 10. Mimari Riskler
### Risk
Capture kötü ise reconstruction çöp olur
### Önlem
Guided short video + quality gates

### Risk
Raw output final sanılır
### Önlem
Validation + registry publish zorunlu

### Risk
GLB/USDZ ile poster/thumbnail farklı sürümlere kayar
### Önlem
Versioned asset package
