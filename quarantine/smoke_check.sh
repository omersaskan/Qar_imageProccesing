#!/bin/bash
set -e

echo "=========================================="
echo "   RUNPOD CUDA RECONSTRUCTION SMOKE CHECK"
echo "=========================================="

# 1. NVIDIA GPU Check
echo -e "\n[1/4] Checking NVIDIA GPU and CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "ERROR: nvidia-smi not found. Ensure you are using a GPU Pod."
    exit 1
fi

# 1.5 Deep Dependency Verification (ldd)
echo -e "\n[1.5/4] Deep Dependency Verification (ldd)..."
# Add OpenMVS bin path to PATH if not already there for ldd check
export PATH="/usr/local/bin:/usr/local/bin/OpenMVS:$PATH"

LDD_BINARIES=("colmap" "InterfaceCOLMAP" "DensifyPointCloud" "ReconstructMesh" "TextureMesh")
for bin in "${LDD_BINARIES[@]}"; do
    BIN_PATH=$(command -v "$bin")
    if [ -z "$BIN_PATH" ]; then
        echo "  Checking $bin... MISSING BINARY"
        exit 1
    fi
    echo -n "  Checking $bin... "
    # Run ldd and check for "not found"
    LDD_OUT=$(ldd "$BIN_PATH" 2>&1)
    if echo "$LDD_OUT" | grep -q "not found"; then
        echo "FAILED"
        echo "ERROR: Missing shared libraries for $bin:"
        echo "$LDD_OUT" | grep "not found"
        echo "Check if libglew2.2, libboost-filesystem1.74.0, libboost-program-options1.74.0, libboost-system1.74.0, libboost-graph1.74.0, libceres2, libglu1-mesa, libopengl0, libgl1 dependecies are installed."
        exit 1
    else
        echo "OK"
    fi
done

# 2. COLMAP Capability Depth Check
echo -e "\n[2/4] Checking COLMAP Binaries & CUDA Linkage..."
if command -v colmap &> /dev/null; then
    COLMAP_VER=$(colmap -h | head -n 1)
    echo "COLMAP Version: $COLMAP_VER"
    
    # Check for dense CUDA capability
    echo "Probing patch_match_stereo (CUDA requirement test)..."
    HELP_OUT=$(colmap patch_match_stereo -h 2>&1)
    if echo "$HELP_OUT" | grep -q "requires CUDA"; then
        echo "CRITICAL FAILURE: COLMAP build reports 'requires CUDA' for dense stage."
        echo "This means the source build lacked CUDA toolkit or linkage failed."
        exit 1
    else
        echo "SUCCESS: patch_match_stereo reports CUDA support."
    fi
    
    # Check for Sift/Feature prefixes
    echo "Mapping flag prefixes..."
    colmap feature_extractor -h | grep -E "Extraction|Matching" | head -n 3
else
    echo "ERROR: colmap binary not found."
    exit 1
fi

# 3. OpenMVS Toolset Check
echo -e "\n[3/4] Checking OpenMVS Tools..."
MVS_TOOLS=("DensifyPointCloud" "ReconstructMesh" "TextureMesh" "InterfaceCOLMAP")
for tool in "${MVS_TOOLS[@]}"; do
    if command -v $tool &> /dev/null; then
        echo "  [OK] $tool"
    else
        echo "  [!!] $tool is MISSING"
        exit 1
    fi
done

# 4. Global Workspace Readiness
echo -e "\n[4/4] Environment Variables..."
echo "RECON_ENGINE_PATH: $RECON_ENGINE_PATH"
echo "OPENMVS_BIN_PATH: $OPENMVS_BIN_PATH"
echo "RECON_PIPELINE: $RECON_PIPELINE"
echo "RECON_USE_GPU: $RECON_USE_GPU"

echo -e "\n=========================================="
echo "      SMOKE CHECK PASSED SUCCESSFULLY"
echo "=========================================="
