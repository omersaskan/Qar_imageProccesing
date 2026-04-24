# RunPod Deployment Guide - CUDA Reconstruction Engine

This guide covers the end-to-end process of building, deploying, and validating the reconstruction engine on a RunPod instance.

## 1. CI/CD Orchestration (GitHub Actions)

The image is automatically built and pushed to **GitHub Container Registry (GHCR)**.

- **Registry**: `ghcr.io`
- **Image Name**: `ghcr.io/omersaskan/qar_imageproccesing` (lowercase)
- **Tags**: `latest` (from main) or `short-sha`

### Required GitHub Secrets
Navigate to `Settings > Secrets and variables > Actions` and ensure the following is available (typically handled by default `GITHUB_TOKEN` but for custom settings):
- No external secrets required if using `GITHUB_TOKEN` permissions defined in the workflow.

### Triggering a Build
- Push any changes to `main` branch.
- **Manual**: Go to `Actions` tab -> `Build and Push Reconstruction Image` -> `Run workflow`.

## 2. Deploying on RunPod

### Pod Setup
1. Log in to [RunPod Console](https://www.runpod.io/console/pods).
2. Click **Deploy**.
3. Select a GPU instance (Recommended: RTX 3090 / 4090 or A6000).
4. **Container Image**: Use your GHCR image (e.g., `ghcr.io/omersaskan/qar_imageproccesing:latest`).
5. **Credentials**: If your registry is private, enter your GitHub Username and a **Personal Access Token (classic)** with `read:packages` scope.
6. **Environment Variables**:
   - `RECON_USE_GPU`: `true`
   - `RECON_PIPELINE`: `colmap_dense`

## 3. Post-Deployment Validation

Once the Pod is running, connect via SSH or Web Terminal and run:

### A. Environment Smoke Check
```bash
./smoke_check.sh
```
This verified CUDA linkage and binary availability.

### B. End-to-End Validation
Run the specific validation script for the `cap_df3eeab8` dataset:
```bash
python3 scratch/validate_cap_df3eeab8.py
```

### C. Inspection
Monitor the logs in real-time:
```bash
tail -f data/reconstructions/VAL_cap_df3eeab8/reconstruction.log
```

## Troubleshooting
- **Failed to parse options**: This usually means a mismatch between the COLMAP version and the flags. The new `ColmapCapabilityManager` handles this automatically.
- **Dense stereo reconstruction requires CUDA**: If `smoke_check.sh` fails here, ensure the Pod was started with compatible NVIDIA drivers and the image build logs show `CUDA_ENABLED=ON`.
- **OpenMVS tools missing**: Check the Docker build logs for `ninja install` failures in the OpenMVS stage.
