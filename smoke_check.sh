#!/bin/bash
set -e

echo "--- RunPod Environment Smoke Check ---"

# 1. GPU Check
echo "Checking NVIDIA GPU status..."
if command -v nvidia-smi &> /dev/null
then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. GPU acceleration may fail."
fi

# 2. COLMAP Capability Check
echo -e "\nChecking COLMAP binary and capabilities..."
if command -v colmap &> /dev/null
then
    COLMAP_PATH=$(command -v colmap)
    echo "COLMAP Found at: $COLMAP_PATH"
    
    echo "Probing feature_extractor help..."
    colmap feature_extractor -h | grep -E "Extraction|Matching|ba_use_gpu" || true
    
    echo "Probing exhaustive_matcher help..."
    colmap exhaustive_matcher -h | grep -E "Extraction|Matching|ba_use_gpu" || true
    
    echo "Probing mapper help..."
    colmap mapper -h | grep -E "Extraction|Matching|ba_use_gpu" || true
else
    echo "ERROR: colmap binary not found in PATH."
    exit 1
fi

# 3. OpenMVS Check
echo -e "\nChecking OpenMVS binaries..."
MVS_TOOLS=("DensifyPointCloud" "ReconstructMesh" "TextureMesh" "InterfaceCOLMAP")
for tool in "${MVS_TOOLS[@]}"; do
    if command -v $tool &> /dev/null; then
        echo "Found OpenMVS tool: $tool"
    else
        echo "WARNING: OpenMVS tool $tool not found."
    fi
done

# 4. Environment Variables
echo -e "\nChecking Environment Variables..."
echo "RECON_ENGINE_PATH: $RECON_ENGINE_PATH"
echo "OPENMVS_BIN_PATH: $OPENMVS_BIN_PATH"
echo "RECON_PIPELINE: $RECON_PIPELINE"
echo "RECON_USE_GPU: $RECON_USE_GPU"

echo -e "\n--- Smoke Check Complete ---"
