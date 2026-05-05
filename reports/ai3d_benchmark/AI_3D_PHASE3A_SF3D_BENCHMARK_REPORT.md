# AI 3D Phase 3A — SF3D Local Benchmark Report

- **Date**: 2026-05-05 00:57:13 UTC
- **Commit SHA**: `a7e9287dfb3b`
- **Environment**: Local Windows/WSL2
- **SF3D Available**: Yes
- **Total Inputs**: 1
- **Total Runs**: 2
- **Successful Runs**: 2
- **Phase Status**: 2 successful SF3D run(s) recorded.

## Results Summary

| Input | Mode | BG | Status | Provider | Duration | GLB Size | Peak VRAM | Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| noisy_background_object.png | balanced | OFF | review | ok | 16.64s | 1.3 MB | 6173.4 MB | None |
| noisy_background_object.png | high | OFF | review | ok | 16.86s | 1.31 MB | 6173.4 MB | None |

## Notes

- This benchmark covers only local SF3D.
- External providers remain disabled and were not touched.
- This is not true multi-view reconstruction.
- Mesh statistics collected via `trimesh` when available.
