# Meshysiz Product Asset Factory

**Meshysiz Product Asset Factory** is a production-grade, modular pipeline for generating high-quality, AR-ready 3D models for e-commerce and product visualization. It transforms guided short videos into cleaned, optimized, and validated digital assets.

---

## 🚀 Key Features
- **Guided Capture & Gating**: Real-time coverage analysis and blur detection to ensure input quality.
- **Automated Reconstruction**: Orchestrated COLMAP and OpenMVS workflows for high-fidelity geometry.
- **Hardened Cleanup Pipeline**: Modular mesh cleaning, decimation, and PBR-ready material normalization.
- **Production-Ready Validation**: Automated quality gates (polycount, texture integrity, alignment) with mandatory approval for edge cases.
- **Atomic Asset Registry**: Secure versioning, audit logging, and atomic "active" pointer management.

---

## 🛠️ Getting Started

### Prerequisites
- **Python 3.11+**
- **FFmpeg** (in PATH)
- **COLMAP & OpenMVS** (Optional for logic tests, required for full reconstruction)

### Installation
```bash
# Clone the repository
git clone https://github.com/omersaskan/Qar_imageProccesing
cd Qar_imageProccesing

# Install core and development dependencies in editable mode
python -m pip install -e ".[dev]"
```

---

## 🧪 Testing & Verification

### Full Test Suite
Run the comprehensive suite (490+ tests) to verify stability:
```bash
python -m pytest
```

### Lightweight Smoke Check
Verify core imports, settings, and API initialization:
```bash
./scripts/smoke_check.sh
```

### Regression Tests
Run specific hardening and production logic tests:
```bash
python -m pytest tests/test_production_hardening.py
```

---

## 📂 Project Structure
```text
.
├── .github/workflows/      # CI/CD pipelines
├── modules/                # Core Business Logic
│   ├── shared_contracts/   # Models, schemas, and errors
│   ├── operations/         # API, Settings, and Worker orchestration
│   ├── capture_workflow/   # Ingestion and quality gating
│   ├── reconstruction/     # 3D reconstruction engine
│   ├── asset_cleanup/      # Mesh & Texture refinement
│   ├── export_pipeline/    # Format conversion (GLB, USDZ)
│   ├── qa_validation/      # Automated quality gates
│   └── asset_registry/     # Versioning and publishing
├── scripts/                # Operational scripts (smoke check, deployment)
├── tests/                  # Integration and regression tests
├── tools/                  # Diagnostic and audit tools
└── ui/                     # Operational dashboard
```

---

## 🛡️ Production Hardening
This repository has been hardened for production readiness:
- **Environment Isolation**: No more machine-specific hardcoded paths; dynamic binary discovery via `shutil.which`.
- **Atomic Registry**: Asset publishing and active pointer updates are now atomic with full audit logs.
- **CORS Security**: Wildcard origins are disabled in Pilot/Production; explicit allowlists are required.
- **Normalized Validation**: Unified status reporting (`pass`/`review`/`fail`) across all validation layers.
- **Clean Hygiene**: Zero tracked logs or scratch data; robust `.gitignore` and CI/CD validation.

---

## 📖 Documentation
- [ARCHITECTURE.md](ARCHITECTURE.MD) - System architecture and data flow.
- [DEPLOYMENT_RUNPOD.md](DEPLOYMENT_RUNPOD.md) - Cloud deployment guide.
- [RUNBOOK.md](RUNBOOK.md) - Operational troubleshooting.

---

## ⚖️ License
Internal proprietary tool for **Meshysiz Team**. 
Contact technical leadership for contribution guidelines.
