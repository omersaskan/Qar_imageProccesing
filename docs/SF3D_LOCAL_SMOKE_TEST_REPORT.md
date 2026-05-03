# SF3D Local Smoke Test Report — Phase 4B

**Date:** 2026-05-03  
**Verdict: FAIL — Windows-native compiler blockers**

---

## Executive Summary

The SF3D local smoke test was executed on a Windows 11 machine with an RTX 5060 Laptop GPU.
The environment was partially prepared (PyTorch + all pure-Python dependencies installed),
but two hard blockers prevent SF3D from running on this platform in its current state:

1. **Compiler blocker** — `Microsoft Visual C++ 14.0 or greater is required` for `texture_baker` and `uv_unwrapper` C++ extensions (mandatory SF3D imports).
2. **Compute capability blocker** — RTX 5060 Laptop is Blackwell (SM_12.0); PyTorch 2.11.0+cu126 does not include PTX for SM_12.0 and will attempt JIT fallback, which is unreliable.

No real inference was run. No output artifacts were produced.

---

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| GPU VRAM | 8.55 GB |
| GPU Compute Capability | **SM_12.0 (Blackwell)** |
| CUDA Driver | 581.95 |
| CUDA Version (driver) | 13.0 |
| Python | 3.11.9 |
| venv path | `external/stable-fast-3d/.venv_sf3d` |

---

## Install Path

```
C:\Users\Lenovo\.gemini\antigravity\scratch\Qar_imageProccesing\
  external\
    stable-fast-3d\          ← git clone --depth 1 Stability-AI/stable-fast-3d
      .venv_sf3d\            ← Python 3.11 venv (created)
  scripts\
    sf3d_worker.py           ← worker entry point (already committed)
```

---

## Commands Executed

```powershell
# 1. Clone
git clone --depth 1 https://github.com/Stability-AI/stable-fast-3d external/stable-fast-3d

# 2. Create venv
py -3.11 -m venv external/stable-fast-3d/.venv_sf3d

# 3. Install PyTorch cu126
.venv_sf3d\Scripts\python.exe -m pip install torch torchvision `
    --index-url https://download.pytorch.org/whl/cu126

# 4. Install pure-Python deps
.venv_sf3d\Scripts\python.exe -m pip install einops==0.7.0 jaxtyping==0.2.31 `
    omegaconf==2.3.0 transformers==4.42.3 open_clip_torch==2.24.0 `
    trimesh==4.4.1 numpy==1.26.4 huggingface-hub==0.23.4 `
    pynanoinstantmeshes==0.0.3 gpytoolbox==0.2.0

# 5. Attempt texture_baker compilation
cd external/stable-fast-3d/texture_baker
.venv_sf3d\Scripts\python.exe setup.py build_ext
```

---

## Diagnostics

### torch diagnostics

```
torch version:           2.11.0+cu126
torch.version.cuda:      12.6
torch.cuda.is_available: True  ⚠ (see warnings below)
GPU name:                NVIDIA GeForce RTX 5060 Laptop GPU
GPU memory:              8.55 GB
```

**CUDA warnings (stderr):**
```
UserWarning: Found GPU0 NVIDIA GeForce RTX 5060 Laptop GPU which is of compute
capability (CC) 12.0.
...
NVIDIA GeForce RTX 5060 Laptop GPU with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities:
  sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90
Please follow the instructions at https://pytorch.org/get-started/locally/
to install a PyTorch release that supports CUDA versions: 12.8, 13.0
```

### pip check (venv)

```
No broken requirements found.
```

### Installed packages (successful)

| Package | Version | Notes |
|---|---|---|
| torch | 2.11.0+cu126 | ⚠ SM_12.0 JIT-only |
| torchvision | 0.26.0 | — |
| einops | 0.7.0 | ✓ |
| jaxtyping | 0.2.31 | ✓ |
| omegaconf | 2.3.0 | ✓ |
| transformers | 4.42.3 | ✓ |
| open-clip-torch | 2.24.0 | ✓ |
| trimesh | 4.4.1 | ✓ |
| numpy | 1.26.4 | ✓ |
| huggingface-hub | 0.23.4 | ✓ |
| pynanoinstantmeshes | 0.0.3 | ✓ (pre-built wheel) |
| gpytoolbox | 0.2.0 | ✓ |

### Not installed (blocked)

| Package | Status | Reason |
|---|---|---|
| `texture_baker` | **FAILED** | Requires MSVC 14.0+ |
| `uv_unwrapper` | **BLOCKED** | Same (not attempted) |
| `rembg[gpu]` | **NOT ATTEMPTED** | Depends on onnxruntime-gpu |
| `sf3d` (package) | **BLOCKED** | Depends on texture_baker/uv_unwrapper |

---

## Blocker Classification

### Blocker 1: Compiler — CRITICAL / BLOCKING

**Error:**
```
error: Microsoft Visual C++ 14.0 or greater is required.
Get it with "Microsoft C++ Build Tools":
https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

**Root cause:** `texture_baker` and `uv_unwrapper` are C++ CUDA extensions built with
`torch.utils.cpp_extension`. On Windows, this requires MSVC (cl.exe). Neither MSVC,
GCC, nor MinGW are installed on this machine.

**Both extensions are mandatory:**
- `sf3d/system.py:33` — `from texture_baker import TextureBaker` — raises ImportError on failure
- `sf3d/models/mesh.py:18` — `from uv_unwrapper import Unwrapper` — raises ImportError on failure

**Consequence:** `import sf3d` fails → worker exits with `sf3d_package_missing` →
API returns `status=unavailable` gracefully (scaffold handles this correctly).

---

### Blocker 2: Compute Capability SM_12.0 — SIGNIFICANT / WORKAROUNDABLE

**Root cause:** RTX 5060 Laptop is NVIDIA Blackwell architecture (SM_12.0).
PyTorch 2.11.0+cu126 was built for SM 5.0–9.0. SM_12.0 is not in the compiled
PTX target list.

**Consequence:** `torch.cuda.is_available()` returns `True` but CUDA kernels will
attempt PTX JIT recompilation, which may fail or produce incorrect results.
GPU acceleration is unreliable with this combination.

---

### Blocker 3: NVCC not installed — SECONDARY

**Root cause:** Only the NVIDIA GPU driver is installed; the CUDA Toolkit (which
includes `nvcc`, CUDA headers, and libraries) is separate and not installed.

**Consequence:** Even with MSVC installed, `texture_baker`'s CUDA extension
(`_C.cu` kernels) would fall back to CPU-only CppExtension (no GPU texture baking).
Full GPU texture baker requires NVCC.

---

## What the Scaffold Did Right

- Worker printed `{"status": "unavailable", "error_code": "sf3d_package_missing"}` and exited 0 when `import sf3d` failed.
- API endpoint returned structured `provider_failure_reason` rather than a 500.
- `SF3DProvider.is_available()` returned `(False, "sf3d_worker_missing")` correctly since the Python path was not yet configured in settings.
- No crash, no stack trace exposed to caller.
- Main application environment was completely untouched.

---

## Artifact Paths

| Artifact | Status |
|---|---|
| `external/stable-fast-3d/` | ✓ Cloned |
| `external/stable-fast-3d/.venv_sf3d/` | ✓ Created |
| `external/stable-fast-3d/.venv_sf3d/Scripts/python.exe` | ✓ Exists |
| `scratch/sf3d_smoke/` | ✓ Created (empty — no inference run) |
| `output.glb` | ✗ Not produced (blocked) |
| `ai3d_manifest.json` | ✗ Not produced (blocked) |

---

## Fix Plan — Windows Native

To unblock on Windows native (in order of application):

1. **Install Microsoft C++ Build Tools 2022** (free)
   ```
   winget install Microsoft.VisualStudio.2022.BuildTools `
     --override "--quiet --add Microsoft.VisualCpp.Tools.HostX64.TargetX64"
   ```

2. **Install CUDA Toolkit 12.6 or 12.8**
   Download: https://developer.nvidia.com/cuda-toolkit-archive
   Required for `nvcc` and CUDA headers (texture_baker GPU extension).

3. **Switch to PyTorch cu128 for Blackwell SM_12.0 support**
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
   PyTorch with cu128 includes SM_12.0 (Blackwell) build targets.

4. **Recompile C++ extensions**
   ```powershell
   cd external/stable-fast-3d/texture_baker && pip install -e .
   cd ../uv_unwrapper && pip install -e .
   ```

5. **Install rembg[gpu]**
   ```powershell
   pip install "rembg[gpu]==2.0.57"
   ```

6. **Install sf3d as editable package**
   ```powershell
   cd external/stable-fast-3d && pip install -e .
   ```

---

## WSL2 Fallback Recommendation

WSL2 (Ubuntu 22.04+) removes all three blockers:

| Issue | Windows Native | WSL2 |
|---|---|---|
| C++ compiler | ✗ MSVC not installed | ✓ GCC 11+ pre-installed |
| NVCC | ✗ Not installed | ✓ Install CUDA Toolkit for WSL2 |
| SM_12.0 (Blackwell) | ⚠ cu128 needed | ✓ Install cu128 wheels |
| texture_baker compile | ✗ BLOCKED | ✓ Expected to work |
| uv_unwrapper compile | ✗ BLOCKED | ✓ Expected to work |

WSL2 CUDA setup:
```bash
# Inside WSL2 Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-8

# Install PyTorch cu128 for Blackwell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install SF3D deps
pip install -r external/stable-fast-3d/requirements.txt
```

---

## Test Suite Status

| Suite | Result |
|---|---|
| `tests/test_ai_3d_generation.py` | 50/50 PASS ✓ |
| `tests/` (full suite) | 722/722 PASS ✓ |

The scaffold is clean — no regressions from the smoke test setup.

---

## Next Step

**Option A — Windows Native (2–3 hours)**
Install MSVC Build Tools + CUDA Toolkit 12.8 → recompile C++ extensions → rerun smoke test.

**Option B — WSL2 (recommended for faster unblock)**
Use existing WSL2 installation, install CUDA for WSL2, install PyTorch cu128,
compile C++ extensions natively. The same `scripts/sf3d_worker.py` worker works
in WSL2 by pointing `SF3D_PYTHON_PATH` to the WSL2 Python binary.
