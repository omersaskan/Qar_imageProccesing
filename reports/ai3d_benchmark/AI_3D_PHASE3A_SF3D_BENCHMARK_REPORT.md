# AI 3D Phase 3A — SF3D Local Benchmark Report

- **Date**: 2026-05-05 01:26:44 UTC
- **Commit SHA**: `8b065f16cd71`
- **Environment**: Local Windows/WSL2
- **SF3D Available**: Yes
- **Total Inputs**: 5
- **Total Runs**: 30
- **Successful Runs**: 30
- **Phase Status**: 30 successful SF3D run(s) recorded.

## Results Summary

| Input | Mode | BG | Status | Provider | Duration | GLB Size | Peak VRAM | Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| noisy_background_object.png | balanced | OFF | review | ok | 22.81s | 1.29 MB | 6173.4 MB | None |
| noisy_background_object.png | balanced | ON | review | ok | 18.97s | 1.29 MB | 6173.4 MB | None |
| noisy_background_object.png | high | OFF | review | ok | 16.67s | 1.31 MB | 6173.4 MB | None |
| noisy_background_object.png | high | ON | review | ok | 19.64s | 1.28 MB | 6173.4 MB | None |
| noisy_background_object.png | ultra | OFF | review | ok | 28.44s | 1.55 MB | 8436.9 MB | None |
| noisy_background_object.png | ultra | ON | review | ok | 26.77s | 1.58 MB | 8459.9 MB | None |
| simple_center_object.png | balanced | OFF | review | ok | 18.02s | 0.98 MB | 6173.1 MB | None |
| simple_center_object.png | balanced | ON | review | ok | 18.31s | 0.97 MB | 6173.1 MB | None |
| simple_center_object.png | high | OFF | review | ok | 18.56s | 0.93 MB | 6173.0 MB | None |
| simple_center_object.png | high | ON | review | ok | 17.2s | 0.96 MB | 6173.0 MB | None |
| simple_center_object.png | ultra | OFF | review | ok | 24.44s | 1.37 MB | 8247.7 MB | None |
| simple_center_object.png | ultra | ON | review | ok | 25.33s | 1.27 MB | 8461.5 MB | None |
| small_offcenter_object.png | balanced | OFF | review | ok | 18.31s | 0.11 MB | 6172.3 MB | None |
| small_offcenter_object.png | balanced | ON | review | ok | 17.27s | 0.1 MB | 6172.3 MB | None |
| small_offcenter_object.png | high | OFF | review | ok | 16.27s | 0.11 MB | 6172.3 MB | None |
| small_offcenter_object.png | high | ON | review | ok | 17.05s | 0.11 MB | 6172.3 MB | None |
| small_offcenter_object.png | ultra | OFF | review | ok | 21.06s | 0.27 MB | 7924.3 MB | None |
| small_offcenter_object.png | ultra | ON | review | ok | 23.09s | 0.27 MB | 7952.2 MB | None |
| tall_object.png | balanced | OFF | review | ok | 17.11s | 0.5 MB | 6172.6 MB | None |
| tall_object.png | balanced | ON | review | ok | 15.24s | 0.51 MB | 6172.6 MB | None |
| tall_object.png | high | OFF | review | ok | 15.22s | 0.52 MB | 6172.6 MB | None |
| tall_object.png | high | ON | review | ok | 14.3s | 0.5 MB | 6172.6 MB | None |
| tall_object.png | ultra | OFF | review | ok | 22.33s | 0.75 MB | 8335.1 MB | None |
| tall_object.png | ultra | ON | review | ok | 24.8s | 0.77 MB | 8277.3 MB | None |
| wide_object.png | balanced | OFF | review | ok | 17.92s | 0.53 MB | 6172.6 MB | None |
| wide_object.png | balanced | ON | review | ok | 17.45s | 0.52 MB | 6172.6 MB | None |
| wide_object.png | high | OFF | review | ok | 18.39s | 0.51 MB | 6172.6 MB | None |
| wide_object.png | high | ON | review | ok | 17.64s | 0.52 MB | 6172.6 MB | None |
| wide_object.png | ultra | OFF | review | ok | 24.61s | 0.74 MB | 8102.5 MB | None |
| wide_object.png | ultra | ON | review | ok | 24.81s | 0.75 MB | 8186.9 MB | None |

## Notes

- This benchmark covers only local SF3D.
- External providers remain disabled and were not touched.
- This is not true multi-view reconstruction.
- Mesh statistics collected via `trimesh` when available.
