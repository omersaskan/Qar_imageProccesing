# SF3D WSL2 Smoke Test Report — Phase 4C

**Date:** 2026-05-03  
**Outcome:** SUCCESS ✅  
**Inference backend:** WSL2 Ubuntu 24.04 + CUDA 12.8 + PyTorch cu128

---

## 1. Environment

| Component | Value |
|-----------|-------|
| OS | WSL2 Ubuntu 24.04 (Windows 11) |
| Python | 3.12.3 (`/home/lenovo/sf3d_venv`) |
| PyTorch | 2.11.0+cu128 |
| CUDA Toolkit | 12.8 (`/usr/local/cuda-12.8`) |
| GPU | NVIDIA GeForce RTX 5060 Laptop (Blackwell SM_12.0) |
| GPU Memory | ~8 GB |
| NVCC | `/usr/local/cuda-12.8/bin/nvcc` |
| HuggingFace model | `stabilityai/stable-fast-3d` |

---

## 2. Install Path

All packages installed into `/home/lenovo/sf3d_venv`.

| Step | Package | Method | Result |
|------|---------|--------|--------|
| 1 | torch 2.11.0+cu128 | `pip install --index-url https://download.pytorch.org/whl/cu128` | ✅ SM_12.0 supported |
| 2 | Pure-Python SF3D deps | `pip install einops jaxtyping omegaconf transformers…` | ✅ |
| 3 | gpytoolbox 0.3.7 | `pip install gpytoolbox` (manylinux wheel, no cmake) | ✅ |
| 4 | texture_baker 0.0.1 | `pip install --no-build-isolation ./texture_baker/` | ✅ CUDA C++ compiled |
| 5 | uv_unwrapper 0.0.1 | `pip install --no-build-isolation ./uv_unwrapper/` | ✅ C++ compiled |
| 6 | rembg 2.0.57 | `pip install rembg==2.0.57` | ✅ |
| 7 | sf3d (editable) | `pip install -e . --no-deps` + added `pyproject.toml` | ✅ |
| 8 | numpy re-pinned | `pip install numpy==1.26.4` | ✅ (transformers compat) |

### Key Fixes Applied

- **`--no-build-isolation`** for texture_baker and uv_unwrapper — pip's isolated build env doesn't see the venv's torch, causing `ModuleNotFoundError: No module named 'torch'`
- **gpytoolbox 0.3.7** instead of 0.2.0 — 0.2.0 has a broken CMake setup; 0.3.7 ships a manylinux wheel
- **`pyproject.toml` added to SF3D root** — repo has no packaging manifest; added minimal `setuptools.build_meta` config
- **numpy pinned back to 1.26.4** — gpytoolbox 0.3.7 pulls in numpy 2.2.6; `transformers==4.42.3` hard-requires `<2.0`

---

## 3. Import Verification

```
texture_baker : OK
uv_unwrapper  : OK
gpytoolbox    : OK
sf3d          : OK
rembg         : OK
ALL_IMPORTS_OK
```

---

## 4. Smoke Inference Run

**Command:**
```bash
/home/lenovo/sf3d_venv/bin/python scripts/sf3d_worker.py \
  --image  scratch/sf3d_smoke/input.png \
  --output-dir /tmp/sf3d_smoke \
  --device cuda \
  --texture-resolution 512 \
  --no-remove-bg
```

**Worker stdout (JSON):**
```json
{
  "status": "ok",
  "output_path": "/tmp/sf3d_smoke/output.glb",
  "output_format": "glb",
  "model_name": "stable-fast-3d",
  "preview_image_path": null,
  "warnings": ["ai_generated_not_true_scan"],
  "metadata": {
    "device": "cuda",
    "input_size": 512,
    "texture_resolution": 512,
    "remesh": "none",
    "pretrained_model": "stabilityai/stable-fast-3d",
    "foreground_ratio": 0.85,
    "peak_mem_mb": 6173.5,
    "output_size_bytes": 1346664
  }
}
```

**Worker stderr (key log lines):**
```
[sf3d_worker] INFO Using device: cuda
[sf3d_worker] INFO Loading SF3D model from: stabilityai/stable-fast-3d
[sf3d_worker] INFO Loaded ViT-B-32 model config.
[sf3d_worker] INFO Loading pretrained ViT-B-32 weights (laion2b_s34b_b79k).
[sf3d_worker] INFO Model loaded on cuda
[sf3d_worker] INFO Image preprocessed: size=(601, 601) mode=RGBA
[sf3d_worker] INFO Running SF3D inference...
[sf3d_worker] INFO Peak GPU memory: 6173.5 MB
[sf3d_worker] INFO GLB exported: /tmp/sf3d_smoke/output.glb  size=1346664 bytes
After Remesh 21350 42696
```

---

## 5. Output Artifact

| Property | Value |
|----------|-------|
| Path (WSL2) | `/tmp/sf3d_smoke/output.glb` |
| Path (Windows) | `scratch/sf3d_smoke/output.glb` |
| Format | glTF Binary v2 (GLB) |
| File size | **1,346,664 bytes (~1.3 MB)** |
| Mesh vertices | 21,350 |
| Mesh faces | 42,696 |
| Peak GPU memory | 6,173.5 MB |
| `file` magic | `glTF binary model, version 2, length 1346664 bytes` |

---

## 6. Acceptance Criteria

| Criterion | Result |
|-----------|--------|
| Model weights accessible (HF gated) | ✅ Authenticated via `huggingface-cli login` |
| All SF3D imports OK | ✅ texture_baker, uv_unwrapper, gpytoolbox, sf3d, rembg |
| Real inference ran on CUDA | ✅ RTX 5060 SM_12.0, no warnings |
| Output file exists | ✅ `/tmp/sf3d_smoke/output.glb` |
| File size > 0 | ✅ 1,346,664 bytes |
| `status: ok` in worker JSON | ✅ |
| Worker contract preserved | ✅ Same `scripts/sf3d_worker.py` interface |

**Overall: PASS ✅**

---

## 7. Phase 4B vs 4C Comparison

| | Phase 4B (Windows-native) | Phase 4C (WSL2) |
|-|--------------------------|-----------------|
| MSVC / GCC | ❌ No MSVC | ✅ GCC 13 |
| NVCC | ❌ Not installed | ✅ CUDA 12.8 |
| SM_12.0 (Blackwell) | ❌ cu126 incompatible | ✅ cu128 fully supported |
| texture_baker compile | ❌ MSVC required | ✅ Compiled successfully |
| uv_unwrapper compile | ❌ MSVC required | ✅ Compiled successfully |
| Inference | ❌ Could not reach | ✅ 1.3 MB GLB produced |

---

## 8. Production Notes

To use SF3D in production (from the main Python env), the `SF3DProvider` subprocess call must:
1. Use `/home/lenovo/sf3d_venv/bin/python` as the interpreter
2. Set env vars: `CUDA_HOME=/usr/local/cuda-12.8`, `LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:...`
3. Set `HF_TOKEN` or ensure `~/.cache/huggingface/token` is populated
4. Note numpy is pinned to `1.26.4` — do not upgrade gpytoolbox without re-checking

### SF3D venv path for SF3DProvider config
```
SF3D_VENV_PYTHON=/home/lenovo/sf3d_venv/bin/python
```
