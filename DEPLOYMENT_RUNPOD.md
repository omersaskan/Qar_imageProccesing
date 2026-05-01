# RunPod Deployment Guide — Meshysiz Asset Factory

End-to-end pod setup, env wiring, volume strategy, SAM2 + AI completion activation,
and post-deploy validation.

---

## 1. Build & Push the Image

CI builds `Dockerfile.runpod` and pushes to GHCR.

- **Registry**: `ghcr.io`
- **Image**: `ghcr.io/<owner>/qar_imageproccesing` (lowercase)
- **Tags**: `latest` (main), `<short-sha>` (pinning)

Triggers: push to `main`, or manual via Actions → *Build and Push Reconstruction Image*.
GitHub default `GITHUB_TOKEN` is sufficient; no extra secrets needed.

Image stages:
1. `nvidia/cuda:11.8.0-devel-ubuntu22.04` builder — COLMAP 3.10 (CUDA 75/80/86/89) + OpenMVS v2.3.0
2. `nvidia/cuda:11.8.0-runtime-ubuntu22.04` runtime — FFmpeg + Python deps + app

---

## 2. Recommended GPU Size

| Profile target | GPU | VRAM | RAM | Suggested instance |
|----------------|-----|------|-----|--------------------|
| `small_*` only | RTX 3090 | 24 GB | 24 GB | runpod cheapest A4000 / 3090 |
| `medium_*` | RTX 4090 | 24 GB | 32 GB | RunPod RTX 4090 (~$0.45/h) |
| `large_*` | A6000 | 48 GB | 50 GB | RunPod A6000 (~$0.79/h) |
| Hunyuan3D-2 lokal | A100 80GB | 80 GB | 117 GB | RunPod A100 (~$1.89/h) |

CPU: 16+ cores recommended (OpenMVS DensifyPointCloud is CPU-heavy).

---

## 3. Volume Mounts (Persistence)

RunPod pods are ephemeral.  Attach a **Network Volume** (or persistent disk)
mounted at `/app/data` so captures, reconstructions, and registry survive
restarts.  Optional mount at `/app/models` for SAM2 checkpoint reuse.

Recommended layout on the volume:
```
/app/data/
   captures/          # uploaded video + extracted frames + masks
   reconstructions/   # COLMAP/OpenMVS scratch + final manifests
   registry/          # AssetPackage state (registry/blobs/, registry/meta/)
   logs/              # rolling app logs
/app/models/sam2/sam2.1_hiera_large.pt   # 898 MB, downloaded once
```

Min volume size:
- 50 GB for small/medium dev
- 200 GB+ for production with multiple concurrent jobs

---

## 4. Pod Configuration

### Web UI (RunPod console)
1. **Container Image**: `ghcr.io/<owner>/qar_imageproccesing:latest`
2. **Registry credentials** (private repos): GitHub user + classic PAT with `read:packages`
3. **Container Disk**: 30 GB (image + scratch)
4. **Volume Mount**: 100 GB+ Network Volume → `/app/data`
5. **Expose Ports**: TCP `8001`
6. **Environment Variables**: copy from `.env.runpod` (see §5)

### Required env vars (minimum)
```
ENV=production
PILOT_API_KEY=<generate strong random key>      # CRITICAL: blocks unauthenticated API access
DATA_ROOT=/app/data
RECON_PIPELINE=colmap_dense
RECON_USE_GPU=true
CORS_ALLOW_ORIGINS=["https://your-frontend.example.com"]
```

Full template: see [`.env.runpod`](.env.runpod) at repo root.

---

## 5. Optional: SAM2 Manual Track Activation

Operators can box-prompt the product on the first frame, and SAM2 video
predictor propagates the mask through every frame (`/sam2_track.html`
in the UI).

1. Set env on the pod:
   ```
   SAM2_ENABLED=true
   SEGMENTATION_METHOD=sam2          # optional — make SAM2 the default mask backend
   SAM2_REVIEW_ONLY=true             # outputs marked for QA review
   ```
2. Install heavy deps once on the pod (NOT baked into image):
   ```bash
   pip install torch==2.4.0 torchvision==0.19.0
   pip install git+https://github.com/facebookresearch/segment-anything-2.git
   ```
3. Restart container — `scripts/download_sam2.sh` auto-fetches the 898 MB
   checkpoint into `/app/models/sam2/sam2.1_hiera_large.pt` (skipped if
   already on the volume).

Verify:
```bash
curl http://localhost:8001/api/health
# Open <pod-url>/sam2_track.html in a browser, paste a session_id, draw a box, hit Track.
```

---

## 6. Optional: AI Completion (Hunyuan3D-2 / Meshy)

For sessions with low surface coverage (forklift bottom, asansör arkası),
generative completion fills unobserved regions under strict policy gates.

**Hunyuan3D-2 via Replicate** (no extra GPU memory — runs on Replicate's infra):
```
AI_3D_PROVIDER=hunyuan3d_replicate
AI_COMPLETION_ENABLED=true
REPLICATE_API_TOKEN=r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Meshy AI**:
```
AI_3D_PROVIDER=meshy
AI_COMPLETION_ENABLED=true
MESHY_API_KEY=msy_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

UI: `<pod-url>/ai_completion.html` — assess shows the decision path
(observed ratio, target status, thresholds), then `Run` invokes the provider.

Quality gates (settings; tune per project):
- `MIN_OBSERVED_SURFACE_FOR_PRODUCTION=0.70` — above this, AI is skipped (already enough)
- `MIN_OBSERVED_SURFACE_FOR_COMPLETION=0.50` — below this and above 30%, run with `review_ready` target
- `MAX_SYNTHESIZED_SURFACE_FOR_PRODUCTION=0.20` — synth > 20% downgrades to review
- `MAX_SYNTHESIZED_SURFACE_FOR_REVIEW=0.50` — synth > 50% downgrades to preview-only

---

## 7. Validation After Boot

Pod startup runs `start_api.sh` → `smoke_check.sh` → `uvicorn`.
Watch the first ~30 seconds:

```bash
docker logs -f <container>
# Expected: 6 SUCCESS lines from smoke_check, then "Uvicorn running on 0.0.0.0:8001".
```

Manual probes:
```bash
curl https://<pod-url>:8001/api/health
curl -H "X-API-Key: $PILOT_API_KEY" https://<pod-url>:8001/api/ready
```

End-to-end smoke: upload a small clip via the UI's `Import New Video Asset`
modal — pick `Object Size: Small`, `Scene Type: On Surface`, `Material: Opaque`.
Watch `/app/data/reconstructions/<job_id>/reconstruction.log` for progress.

---

## 8. Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `Failed to parse options` (COLMAP) | COLMAP version mismatch — `ColmapCapabilityManager` handles automatically; if persists, rebuild image |
| `Dense stereo reconstruction requires CUDA` | Pod started without `--gpus all` or driver mismatch — check `nvidia-smi` inside container |
| `OpenMVS tools missing` | OpenMVS build failure in CI — re-trigger build, watch ninja install logs |
| `FFmpeg failed: Invalid data` | Source video corrupt or codec unsupported — try re-encoding with `ffmpeg -c:v libx264` |
| `OOM: CUDA out of memory` (PatchMatchStereo) | `RECON_MAX_IMAGE_SIZE` too high for VRAM. RTX 5060 8GB → 1800; RTX 4090 24GB → 3000; A6000 48GB → 4000 |
| `MemoryError` in cleaner.py | Mesh exceeds `RECON_PRE_CLEANUP_TARGET_FACES` and Python decimation OOMs — bump RAM or lower preset |
| SAM2 wrapper reports unavailable | torch/sam2 not installed (see §5 step 2), or checkpoint download failed (run `scripts/download_sam2.sh`) |
| AI completion endpoint 503 | Provider unavailable: missing `REPLICATE_API_TOKEN` / `MESHY_API_KEY`, or `replicate`/`requests` package not installed |
| All uploads return 500 | `ENV=production` requires `PILOT_API_KEY` set + ffmpeg/colmap on PATH; check `/api/ready` |

---

## 9. Cost & Scaling Notes

- A single reconstruction on RTX 4090 costs roughly **$0.10–0.20** (5–25 min).
- Hunyuan3D-2 via Replicate: **$0.05–0.15 per completion** (no pod GPU spend).
- For batch processing, run pod with `MESHYSIZ_EMBEDDED_WORKER=true` and queue
  uploads via `POST /api/sessions/upload`.
- For high-throughput, separate API pod (`UVICORN_WORKERS=4`,
  `MESHYSIZ_EMBEDDED_WORKER=false`) and worker pod sharing the same volume.
