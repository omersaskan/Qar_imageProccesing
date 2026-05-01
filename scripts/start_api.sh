#!/bin/bash
# Production API entrypoint for RunPod / Docker.
# Runs the smoke check first, then starts uvicorn on 0.0.0.0:8001.

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "  QAR / Meshysiz Asset Factory — STARTUP"
echo "=========================================="
echo "Env:               ${ENV:-unknown}"
echo "RECON_PIPELINE:    ${RECON_PIPELINE:-unset}"
echo "RECON_USE_GPU:     ${RECON_USE_GPU:-unset}"
echo "CAPTURE_PROFILE:   ${CAPTURE_PROFILE:-unset}"
echo "AI_3D_PROVIDER:    ${AI_3D_PROVIDER:-none}"
echo "SAM2_ENABLED:      ${SAM2_ENABLED:-false}"
echo "DATA_ROOT:         ${DATA_ROOT:-data}"
echo "=========================================="

# Optional SAM2 model fetch (no-op if checkpoint exists or SAM2 disabled)
if [ "${SAM2_ENABLED:-false}" = "true" ]; then
    if [ -f "scripts/download_sam2.sh" ]; then
        echo "[startup] SAM2 enabled — ensuring checkpoint is present..."
        ./scripts/download_sam2.sh || echo "[startup] SAM2 download non-fatal: wrapper will report unavailable."
    fi
fi

# Quick health check before binding port
./scripts/smoke_check.sh

# Bind to 0.0.0.0 so RunPod port-forwarding works.
echo ""
echo "[startup] Launching uvicorn on 0.0.0.0:8001 ..."
exec python3 -m uvicorn modules.operations.api:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers "${UVICORN_WORKERS:-1}" \
    --log-level "${UVICORN_LOG_LEVEL:-info}"
