# Regression Reconciliation Phase 2 Plan

Objective: Eliminate the remaining 38 test failures across the full suite while strictly maintaining production quality gates.

## Priority 1: Validator Hardening & Reconciliation
**Goal**: Stabilize `AssetValidator` behavior for profile-aware missing metrics and correctly handle decimation/structure flags.

- [x] **Inventory Failures**: Categorized 38 failures into Validator, Upload, Integration, and Sprint5 groups.
- [ ] **Profile-Aware Filtering Check**: Update `validate_object_filtering` to fail/review missing `filtering_status` for mobile/desktop profiles, while allowing it for `raw_archive`.
- [ ] **Structural Ready Logic**: Ensure `export_delivery_gate` correctly evaluates `structural_export_ready` or `delivery_ready` based on actual GLB contents.
- [ ] **Texture QA Precedence**: Missing texture metrics must fail/review for textured delivery profiles.
- [ ] **Decimation Precedence**: Implement failure precedence logic in `validate_decimation`: `failed_visual_integrity` > `failed_uv_preservation`.

## Priority 2: Upload Preflight & Profile Logic
**Goal**: Reorder preflight checks to ensure correct root cause reporting and allow profile-based duration overrides.

- [ ] **Dynamic Duration Gate**: Allow `min_video_duration_sec` to be overridden via `CaptureProfile` or test-specific settings, preventing 15s gate from masking other features in 5s video tests.
- [ ] **Error Order Reconciliation**:
    1. JSON/Manifest Parse Errors
    2. Profile/Metadata Errors
    3. AR Quality Gates
    4. Video Physical Validation (Duration, FPS, Resolution)
- [ ] **Effective Settings**: Ensure `api.py` consistently uses `eff_settings` derived from the selected `CaptureProfile`.

## Priority 3: Full Suite Stabilization
**Goal**: Resolve integration-level regressions caused by metadata structure changes.

- [ ] **E2E Smoke Fixes**: Resolve `KeyError` in `test_pilot_smoke_e2e.py` by ensuring all hardened validation keys are present in the final report.
- [ ] **Sprint 5C & Retry**: Stabilize decimation and texturing retry tests by aligning them with new path structures and budget limits.
- [ ] **StopIteration Fix**: Ensure texture retry loops handle empty candidate lists gracefully.

## Verification Plan
- **Validator Group**: `py -m pytest tests/qa_validation tests/test_part4_verification.py tests/test_delivery_hardening.py -q`
- **Upload Group**: `py -m pytest tests/test_api_upload_preflight.py tests/test_fix_regressions.py tests/test_hardened_upload.py -q`
- **Full Suite**: `py -m pytest tests -q`
