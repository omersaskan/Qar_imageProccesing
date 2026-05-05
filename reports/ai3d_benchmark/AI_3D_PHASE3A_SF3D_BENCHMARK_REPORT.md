# AI 3D Phase 3A — SF3D Local Benchmark Report

- **Date**: 2026-05-05 00:49:06 UTC
- **Commit SHA**: `165cc09c3efc`
- **Environment**: Local Windows/WSL2
- **SF3D Available**: Yes
- **Total Inputs**: 2
- **Total Runs**: 4
- **Successful Runs**: 4
- **Phase Status**: 4 successful SF3D run(s) recorded.

## Results Summary

| Input | Mode | BG | Status | Provider | Duration | GLB Size | Peak VRAM | Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| noisy_background_object.png | balanced | OFF | review | ok | 20.03s | 1.29 MB | 6173.4 MB | None |
| noisy_background_object.png | balanced | ON | review | ok | 15.95s | 1.3 MB | 6173.4 MB | None |
| simple_center_object.png | balanced | OFF | review | ok | 15.53s | 0.98 MB | 6173.1 MB | None |
| simple_center_object.png | balanced | ON | review | ok | 17.2s | 0.97 MB | 6173.1 MB | None |

## Notes

- This benchmark covers only local SF3D.
- External providers remain disabled and were not touched.
- This is not true multi-view reconstruction.
- Mesh statistics collected via `trimesh` when available.
