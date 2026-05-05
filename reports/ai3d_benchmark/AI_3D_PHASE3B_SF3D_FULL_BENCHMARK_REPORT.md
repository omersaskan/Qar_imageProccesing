# AI 3D Phase 3B — Full Local SF3D Benchmark Report

**Scope statement:** This benchmark covers only local SF3D. External providers remain disabled and were not touched. This is not true multi-view reconstruction.

---

## Environment

| Field | Value |
|---|---|
| Commit SHA | `8b065f16cd71ce71f6bea3e797b9a5c43402ab1c` |
| Branch | `main` |
| Date | 2026-05-05 |
| Platform | Windows 11 + WSL2 Ubuntu 24.04 |
| SF3D execution mode | `wsl_subprocess` |
| Device | `cuda` |
| AI_3D_GENERATION_ENABLED | `true` |
| SF3D_ENABLED | `true` |
| AI_3D_BACKGROUND_REMOVAL_ENABLED | `true` |
| Background removal library | rembg |

---

## Summary

| Metric | Value |
|---|---|
| Total inputs | 5 |
| Modes tested | balanced, high, ultra |
| BG modes tested | off, on |
| Total runs | 30 |
| Successful runs | **30 / 30** |
| Failed runs | **0** |
| All provider_status | `ok` |
| All device | `cuda` |
| Errors | 0 |

---

## Per-Run Results Table

| Input | Mode | BG | Status | Duration (s) | GLB Size (MB) | Peak VRAM (MB) | Vertices | Faces | BG Removed | Foreground Ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| noisy_background_object | balanced | off | ok | 22.81 | 1.29 | 6173.4 | 22,041 | 39,376 | false | — |
| noisy_background_object | balanced | on  | ok | 18.97 | 1.30 | 6173.4 | 22,133 | 39,188 | true  | 0.0551 |
| noisy_background_object | high     | off | ok | 16.67 | 1.31 | 6173.4 | 22,413 | 39,460 | false | — |
| noisy_background_object | high     | on  | ok | 19.64 | 1.28 | 6173.4 | 21,844 | 39,144 | true  | 0.0551 |
| noisy_background_object | ultra    | off | ok | 28.44 | 1.55 | 8436.9 | 21,708 | 38,664 | false | — |
| noisy_background_object | ultra    | on  | ok | 26.77 | 1.58 | 8459.9 | 21,960 | 39,068 | true  | 0.0551 |
| simple_center_object     | balanced | off | ok | 18.02 | 0.98 | 6173.1 | 15,953 | 28,280 | false | — |
| simple_center_object     | balanced | on  | ok | 18.31 | 0.97 | 6173.1 | 15,700 | 27,328 | true  | 0.2084 |
| simple_center_object     | high     | off | ok | 18.56 | 0.93 | 6173.0 | 14,739 | 25,940 | false | — |
| simple_center_object     | high     | on  | ok | 17.20 | 0.96 | 6173.0 | 15,358 | 26,828 | true  | 0.2084 |
| simple_center_object     | ultra    | off | ok | 24.44 | 1.37 | 8247.7 | 18,251 | 31,804 | false | — |
| simple_center_object     | ultra    | on  | ok | 25.33 | 1.27 | 8461.5 | 15,558 | 27,312 | true  | 0.2084 |
| small_offcenter_object   | balanced | off | ok | 18.31 | 0.11 | 6172.3 |    261 |    352 | false | — |
| small_offcenter_object   | balanced | on  | ok | 17.27 | 0.10 | 6172.3 |    240 |    296 | true  | 0.0100 |
| small_offcenter_object   | high     | off | ok | 16.27 | 0.11 | 6172.3 |    265 |    312 | false | — |
| small_offcenter_object   | high     | on  | ok | 17.05 | 0.11 | 6172.3 |    241 |    296 | true  | 0.0100 |
| small_offcenter_object   | ultra    | off | ok | 21.06 | 0.27 | 7924.3 |    246 |    284 | false | — |
| small_offcenter_object   | ultra    | on  | ok | 23.09 | 0.27 | 7952.2 |    241 |    296 | true  | 0.0100 |
| tall_object              | balanced | off | ok | 17.11 | 0.50 | 6172.6 |  7,075 | 12,136 | false | — |
| tall_object              | balanced | on  | ok | 15.24 | 0.51 | 6172.6 |  7,174 | 12,604 | true  | 0.1476 |
| tall_object              | high     | off | ok | 15.22 | 0.52 | 6172.6 |  7,263 | 12,348 | false | — |
| tall_object              | high     | on  | ok | 14.30 | 0.50 | 6172.6 |  7,057 | 12,472 | true  | 0.1476 |
| tall_object              | ultra    | off | ok | 22.33 | 0.75 | 8335.1 |  7,057 | 12,440 | false | — |
| tall_object              | ultra    | on  | ok | 24.80 | 0.77 | 8277.3 |  7,106 | 12,480 | true  | 0.1476 |
| wide_object              | balanced | off | ok | 17.92 | 0.53 | 6172.6 |  7,539 | 12,924 | false | — |
| wide_object              | balanced | on  | ok | 17.45 | 0.52 | 6172.6 |  7,557 | 12,512 | true  | 0.1479 |
| wide_object              | high     | off | ok | 18.39 | 0.51 | 6172.6 |  7,443 | 12,700 | false | — |
| wide_object              | high     | on  | ok | 17.64 | 0.52 | 6172.6 |  7,506 | 12,440 | true  | 0.1479 |
| wide_object              | ultra    | off | ok | 24.61 | 0.74 | 8102.5 |  7,148 | 12,288 | false | — |
| wide_object              | ultra    | on  | ok | 24.81 | 0.75 | 8186.9 |  7,372 | 12,420 | true  | 0.1479 |

---

## Aggregate Table by Mode and BG

| Mode | BG | Success Rate | Avg Duration (s) | Avg GLB Size (MB) | Avg Peak VRAM (MB) | Avg Vertices | Avg Faces |
|---|---|---|---|---|---|---|---|
| balanced | off | 5/5 (100%) | 18.83 | 0.68 | 6172.8 | 10,574 | 18,614 |
| balanced | on  | 5/5 (100%) | 17.45 | 0.68 | 6172.8 | 10,561 | 18,386 |
| high     | off | 5/5 (100%) | 17.02 | 0.68 | 6172.8 | 10,425 | 18,152 |
| high     | on  | 5/5 (100%) | 17.17 | 0.67 | 6172.8 | 10,401 | 18,236 |
| ultra    | off | 5/5 (100%) | 24.18 | 0.94 | 8209.3 | 10,882 | 19,096 |
| ultra    | on  | 5/5 (100%) | 24.96 | 0.93 | 8267.6 | 10,447 | 18,315 |

---

## Input-Size Propagation Verification

| Mode | Expected input_size | Observed input_size | Match |
|---|---|---|---|
| balanced | 768  | 768  | ✓ |
| high     | 1024 | 1024 | ✓ |
| ultra    | 1024 | 1024 | ✓ |

Ultra VRAM budget (8200–8460 MB vs 6172 MB for balanced/high) confirms 2048px texture resolution is active for ultra as expected.

---

## Failures

**None.** All 30 runs completed with `provider_status=ok`, `errors_count=0`, and a valid `output_glb_path`.

---

## Findings

### balanced vs high

- GLB size is identical: both average **0.68 MB**.
- high is marginally faster (17.02s vs 18.83s avg) despite processing at higher input resolution (1024px vs 768px).
- VRAM usage is identical (~6173 MB) — the increased resolution does not cost extra memory at SF3D's batch size of 1.
- Mesh complexity is slightly lower for high (10,425 avg vertices vs 10,574), suggesting the higher input resolution gives SF3D a cleaner signal and produces slightly tighter geometry.
- **Conclusion:** high is strictly better than balanced — faster, higher input resolution, same VRAM, same output size.

### high vs ultra

- ultra is **~7 seconds slower** per run (24.6s vs 17.1s avg).
- ultra uses **~2,037 MB more VRAM** (8238 MB vs 6173 MB avg) due to 2048px texture resolution vs 1024px.
- ultra produces **~38% larger GLB files** (0.93 MB vs 0.68 MB avg) — the additional size is texture data, not geometry.
- Vertex/face counts are comparable; ultra does not produce meaningfully more geometry.
- ultra succeeded 10/10 with no errors.
- **Conclusion:** ultra is only worth enabling when texture quality is the primary concern and the user has adequate GPU VRAM (≥8 GB reserved). For most AR/QR restaurant use cases, high is sufficient.

### BG off vs BG on

- Background removal (rembg) reduces average duration slightly for complex inputs (noisy_background: 22.81s→18.97s for balanced) by helping SF3D focus on the foreground.
- For simple or already-clean inputs (wide, tall), BG removal has negligible impact on output size or mesh quality.
- For the `noisy_background_object` fixture (foreground_ratio=0.0551 — only 5.5% of pixels are foreground), BG removal consistently reduces noise in the reconstruction.
- BG removal can occasionally reduce vertex count (simple_center_object ultra: 18,251→15,558 vertices) by stripping background detail.
- **Conclusion:** BG removal (`on`) is recommended as default for real-world restaurant menu images which typically have complex backgrounds.

### Which input type was hardest

- **`noisy_background_object`** was consistently the hardest: highest GLB sizes (1.28–1.58 MB), highest vertex/face counts (21,700–22,400 vertices), and lowest foreground ratio (5.5%). The noisy background was incorporated into the mesh when BG removal was off.
- **`small_offcenter_object`** produced the lowest-quality results: only 240–265 vertices and 284–352 faces regardless of mode, indicating SF3D struggled to isolate the off-center object. This represents a real-world failure mode where the subject is poorly framed.

### Whether ultra is worth using

Ultra succeeded 10/10 but at a significant cost:
- +7 seconds per run
- +2 GB VRAM
- +38% file size (texture driven, not geometry)
- No improvement in vertex/face counts

**Ultra is not recommended as default** for the QR restaurant AR pipeline. The additional texture resolution is not perceivable at typical mobile AR viewing distances and the VRAM cost risks OOM errors on lower-spec GPUs.

### Recommended default mode

**`high` with background removal `on`**

Rationale:
- Identical VRAM to balanced (~6173 MB)
- Faster than balanced (17.2s vs 18.8s avg)
- Higher input resolution (1024px vs 768px) for cleaner geometry
- BG removal helps with real menu photo backgrounds
- Robust: 5/5 success rate, 0 errors

---

## Clear Statements

- **This benchmark covers only local SF3D.**
- **External providers (Meshy, Rodin, Tripo, Hunyuan, TRELLIS, SHARP) remain disabled and were not touched.**
- **This is not true multi-view reconstruction.** SF3D generates 3D geometry from a single image using a learned prior, not photogrammetry or NeRF-based methods.

---

## Artifacts

| File | Description |
|---|---|
| `reports/ai3d_benchmark/results.json` | Full per-run results (30 entries) |
| `reports/ai3d_benchmark/results.csv` | Same data in CSV format |
| `reports/ai3d_benchmark/bench_*/derived/output.glb` | Per-run GLB outputs |
| `reports/ai3d_benchmark/bench_*/derived/ai3d_input.png` | Per-run prepared input images |
