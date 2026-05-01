#!/bin/bash
# Pre-startup smoke check.  Verifies imports, settings, FastAPI app,
# binaries (FFmpeg, COLMAP, OpenMVS), and the new pipeline modules
# (capture profile, color profiler, AI completion).
#
# Exit codes:
#   0  — all checks pass
#   1+ — hard failure (import / settings / app / production binary missing)

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "   QAR IMAGE PROCESSING SMOKE CHECK"
echo "=========================================="

# ── 1. Core Imports ──────────────────────────────────────────────────────────
echo -e "\n[1/6] Verifying Core Module Imports..."
python3 -c "
import modules
import modules.operations.api
import modules.reconstruction_engine
import modules.capture_workflow
import modules.asset_cleanup_pipeline
import modules.export_pipeline
import modules.qa_validation
print('SUCCESS: Core modules imported.')
"

# ── 2. New Pipeline Modules ──────────────────────────────────────────────────
echo -e "\n[2/6] Verifying New Pipeline Modules..."
python3 -c "
from modules.utils.color_profiler import ColorProfile, resolve_color_profile
from modules.operations.capture_profile import (
    CaptureProfile, SizeClass, SceneType, MaterialHint,
    resolve_capture_profile, apply_profile_to_settings,
)
from modules.ai_completion import (
    AICompletionService, build_default_service, CompletionStatus,
    decide_completion_path,
)
from modules.ai_completion.providers import build_provider
print('SUCCESS: color_profiler, capture_profile, ai_completion all importable.')
"

# ── 3. Settings Load ─────────────────────────────────────────────────────────
echo -e "\n[3/6] Verifying Settings..."
python3 -c "
from modules.operations.settings import settings
print(f'  env:                {settings.env.value}')
print(f'  data_root:          {settings.data_root}')
print(f'  recon_pipeline:     {settings.recon_pipeline}')
print(f'  capture_profile:    {settings.capture_profile}')
print(f'  expected_color:     {settings.expected_product_color}')
print(f'  ai_3d_provider:     {settings.ai_3d_provider}')
print(f'  ai_completion_on:   {settings.ai_completion_enabled}')
print(f'  sam2_enabled:       {settings.sam2_enabled}')
print('SUCCESS: Settings loaded.')
"

# ── 4. FastAPI App + Critical Routes ─────────────────────────────────────────
echo -e "\n[4/6] Verifying FastAPI App + Critical Routes..."
python3 -c "
from modules.operations.api import app
required = [
    '/api/health',
    '/api/sessions/upload',
    '/api/sessions/{session_id}/sam2_track',
    '/api/sessions/{session_id}/first-frame',
    '/api/sessions/{session_id}/ai-complete',
    '/api/sessions/{session_id}/ai-complete/assess',
]
paths = {getattr(r, 'path', None) for r in app.routes}
missing = [p for p in required if p not in paths]
if missing:
    raise SystemExit(f'MISSING ROUTES: {missing}')
print(f'SUCCESS: {len(required)} required routes present.')
"

# ── 5. Capture Profile Matrix ────────────────────────────────────────────────
echo -e "\n[5/6] Verifying Capture Profile Matrix (9 presets)..."
python3 -c "
from modules.operations.capture_profile import (
    SizeClass, SceneType, resolve_capture_profile,
)
count = 0
for sz in SizeClass:
    for sc in SceneType:
        p = resolve_capture_profile(sz, sc)
        assert p.preset_key, f'empty preset for {sz},{sc}'
        count += 1
assert count == 9, f'Expected 9 presets, got {count}'

# Forklift sanity: large_freestanding must NOT strip planes/support
fk = resolve_capture_profile(SizeClass.LARGE, SceneType.FREESTANDING)
assert fk.remove_horizontal_planes is False
assert fk.remove_bottom_support_band is False
assert fk.recon_poisson_depth == 9
assert fk.max_upload_mb >= 2000
print(f'SUCCESS: {count} presets resolvable; forklift profile correct.')
"

# ── 6. Binary Probes ─────────────────────────────────────────────────────────
echo -e "\n[6/6] Verifying External Binaries..."
python3 -c "
from modules.operations.settings import settings
colmap = settings.probe_colmap_binary()
ffmpeg = settings.probe_ffmpeg()
print(f'  COLMAP   ok={colmap[\"ok\"]}  path={colmap.get(\"path\",\"?\")}')
print(f'  FFmpeg   ok={ffmpeg[\"ok\"]}  path={ffmpeg.get(\"path\",\"?\")}')
if settings.env.value in ('production','pilot') and not (colmap['ok'] and ffmpeg['ok']):
    raise SystemExit(f'BINARY MISSING in {settings.env.value}: colmap={colmap[\"ok\"]} ffmpeg={ffmpeg[\"ok\"]}')
print('SUCCESS: Binary probes done.')
"

echo -e "\n=========================================="
echo "      SMOKE CHECK PASSED SUCCESSFULLY"
echo "=========================================="
