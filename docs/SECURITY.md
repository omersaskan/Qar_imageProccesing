# Security Guidelines & Hardening

This document outlines the security measures implemented in the Meshysiz Asset Factory pipeline to ensure production-grade stability and data integrity.

## API Security

### API Key Enforcement
- All sensitive endpoints require an `X-API-Key` header.
- The key is configured via the `PILOT_API_KEY` environment variable.
- In `local_dev` mode, the key check is optional for easier local development.

### CORS Policy
- CORS is restricted via `CORS_ALLOW_ORIGINS` in `.env`.
- Production deployments should set this to the specific mobile app domain or origin.
- Defaults to `["*"]` for compatibility, but hardening is recommended for production.

### Path Safety & Identifier Validation
- All user-provided identifiers (`product_id`, `operator_id`, `session_id`) are sanitized using `validate_identifier()`.
- This prevents directory traversal attacks and ensures filesystem compatibility.
- Identifiers must be alphanumeric (allowing hyphens and underscores).

## Video Ingestion Hardening

### Normalization Pipeline
- Every uploaded video is normalized via FFmpeg to a strictly defined format:
  - **Codec**: H.264 (libx264)
  - **Pixel Format**: yuv420p
  - **Container**: MP4
- This ensures that downstream CV2/COLMAP tasks never encounter incompatible streams (e.g., variable frame rate WebM, high-bit-depth MOV).

### Quality Gating
- Uploads are rejected if they do not meet the minimum quality thresholds:
  - Minimum Duration: 15.0 seconds
  - Minimum FPS: 20.0
  - Integrity: Must be readable by OpenCV and contain valid frames.

## Reconstruction & Texturing

### Atomic Session Management
- Sessions are managed via `SessionManager` with `FileLock` to prevent race conditions during state updates.
- Failed uploads are marked `FAILED` or cleaned up immediately to prevent disk pollution.

### Process Isolation & Timeouts
- External binaries (COLMAP, OpenMVS) are run with strict timeouts.
- If a process hangs, it is killed, and its partial logs are preserved for diagnostics.
- This prevents single-session failures from blocking the entire worker pool.

## Frontend Security

### XSS Mitigation
- The dashboard uses `textContent` and DOM element construction instead of `innerHTML` for all user-controlled data.
- API strings and guidance messages are escaped before rendering.

### Secure Context
- The mobile AR interface enforces `window.isSecureContext` (HTTPS) to ensure camera access is granted and data is transmitted securely.
