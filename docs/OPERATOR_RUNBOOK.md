# Meshysiz Asset Factory - Operator Runbook (Phase 3)

This document provides operational guidance for deploying and managing the 3D Asset Pipeline in pilot and production environments.

## 1. Environment Configuration

The system recognizes three profiles controlled by the `ENV` environment variable:
- `local_dev`: Default. Relaxed security, verbose logs, stubs allowed.
- `pilot`: Hardened. Structured JSON logs (INFO level), API Key required.
- `production`: Fully hardened. Structured JSON logs (WARNING level), API Key mandatory.

### Required Environment Variables (.env)
```bash
# Core
ENV=pilot # local_dev | pilot | production
DATA_ROOT=data
PILOT_API_KEY=your_secure_random_key_here

# Binaries
RECON_ENGINE_PATH=C:\colmap\colmap.exe
OPENMVS_BIN_PATH=C:\openmvs\bin

# Worker Cadence
WORKER_INTERVAL_SEC=5
```

## 2. API Authentication

In `pilot` and `production` modes, all sensitive endpoints require the `X-API-KEY` header.

| Header | Description | Required In |
| --- | --- | --- |
| `X-API-KEY` | Static key matching `PILOT_API_KEY` | Pilot, Production |

**Public Endpoints:**
- `GET /api/health`: Always public. Returns status and environment.

**Protected Endpoints:**
- `GET /api/ready`: Checks if binaries and directories are correctly mounted.
- `POST /api/sessions/upload`: Core ingestion.
- `GET /api/sessions/{id}/guidance`: Operator feedback.

## 3. Operational Monitoring

### Health Checks
- **Liveness**: `GET /api/health`
- **Readiness**: `GET /api/ready` (Returns 200 with `status: "ready"` or `"not_ready"` + issues).

### Observability
Logs are stored in `data/logs/factory.log` in JSON format.
In Pilot/Prod, logs include:
- `timestamp`: UTC ISO8601
- `level`: INFO, WARNING, ERROR
- `stage`: Current pipeline stage (e.g., `extraction`, `reconstruction`)
- `duration_ms`: Execution time for specific stages.
- `env`: The active profile.

## 4. Retention Policy

The `IngestionWorker` runs a periodic cleanup service every hour.

| Artifact Type | Retention (Success) | Retention (Fail/Review) |
| --- | --- | --- |
| **Raw Video/Frames** | 3 Days | 14 Days |
| **Recon Scratch** | 48 Hours | 48 Hours |
| **Final GLB** | Permanent | N/A |
| **Manifests/Reports** | Permanent | Permanent |
| **Audit History** | Permanent | Permanent |

## 5. Python Dependency Verification

To ensure production-grade assets, the system requires specific Python libraries. Missing dependencies will cause the `/api/ready` check to fail and will block ingestion in `pilot` or `production` environments.

### Required Packages
- **rembg**: ML-first object segmentation.
- **onnxruntime**: Inference engine for segmentation models (CPU-first).
- **fast-simplification**: High-performance mesh decimation.

### Installation Guidance (CPU-First)
Run the following commands in the application's Python environment:
```powershell
# Standard installation (CPU)
pip install rembg onnxruntime fast-simplification
```

### Optional GPU Acceleration
If a compatible NVIDIA GPU is available, you may use the GPU-optimized inference engine instead of the standard `onnxruntime`:
```powershell
# Optional: Replace onnxruntime with GPU support
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

## 6. Troubleshooting Common Failures

### 1. `not_ready` status in `/api/ready`
- **Cause**: Incorrect `RECON_ENGINE_PATH`, `DATA_ROOT`, or missing Python dependencies.
- **Action**: Check `.env` paths first. If structural checks pass, check the `dependencies` object in the API response and install missing packages as shown in Section 5.

### 2. `503 Service Unavailable` on Upload
- **Cause**: "System Environment Incomplete". One or more critical dependencies are missing in a non-dev environment.
- **Action**: Ingestion is blocked to prevent degraded quality. Install required dependencies (Section 5) and restart the service.

### 3. `unauthorized` (401) errors
- **Cause**: Missing or mismatching `X-API-KEY` header.
- **Action**: Verify the key in `.env` and the client header.

### 4. Session stuck in `CREATED`
- **Cause**: IngestionWorker not running or crashed.
- **Action**: Check `GET /api/worker/status`. Verify local filesystem locks in `data/worker.process`.

### 5. Recon Scratch Missing
- **Cause**: Retention policy pruned the heavy intermediate data after 48 hours.
- **Action**: Normal behavior. Final GLBs and manifests are preserved.
