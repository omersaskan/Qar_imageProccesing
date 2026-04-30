#!/bin/bash
set -e

echo "=========================================="
echo "   QAR IMAGE PROCESSING SMOKE CHECK"
echo "=========================================="

# 1. Package Import Check
echo -e "\n[1/4] Verifying Package Imports..."
python3 -c "import modules; import modules.operations.api; import modules.reconstruction_engine; print('SUCCESS: Core modules imported.')"

# 2. Settings Load Check
echo -e "\n[2/4] Verifying Settings Loading..."
python3 -c "from modules.operations.settings import settings; print(f'SUCCESS: Settings loaded for env: {settings.env.value}')"

# 3. FastAPI App Import Check
echo -e "\n[3/4] Verifying FastAPI App Initialization..."
python3 -c "from modules.operations.api import app; print('SUCCESS: FastAPI app initialized.')"

# 4. Binary Probe Graceful Failure Check
echo -e "\n[4/4] Verifying Binary Probes (Graceful Failure)..."
python3 -c "from modules.operations.settings import settings; colmap = settings.probe_colmap_binary(); print(f'COLMAP Probe OK: {colmap[\"ok\"]} (Expected in dev/smoke if no binary)'); ffmpeg = settings.probe_ffmpeg(); print(f'FFmpeg Probe OK: {ffmpeg[\"ok\"]} (Expected in dev/smoke if no binary)')"

echo -e "\n=========================================="
echo "      SMOKE CHECK PASSED SUCCESSFULLY"
echo "=========================================="
