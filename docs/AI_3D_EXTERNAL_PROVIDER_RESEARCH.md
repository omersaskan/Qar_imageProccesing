# AI 3D External Provider Research

This document evaluates external and future internal providers for the AI 3D generation pipeline.

## Provider Comparison Matrix

| Provider | Expected Quality | Geometry Strength | Texture/PBR Strength | Multi-image Support | Output Formats | Topology Controls | API Complexity | Self-hosted Support | Hardware Needs | License/Privacy Risk | Recommended Phase |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SF3D** | High (Fast) | Good (Tri) | High (Unlit/PBR) | No | GLB | Low | Yes (Local/WSL) | 24GB VRAM | Low (Apache 2.0) | Phase 1 (Baseline) |
| **Rodin (Gen-2)** | Premium+ | Extreme (Quad) | Extreme (PBR) | Yes | GLB, OBJ, FBX | High (Quad/Tri) | Moderate (Async) | No | Cloud | Moderate (Commercial) | Phase 1.5 (Premium) |
| **Meshy** | High | High (Quad) | High (PBR) | Yes | GLB, OBJ, FBX | High (Quad/Tri) | Moderate (Async) | No | Cloud | Moderate (Commercial) | Phase 1.5 (All-around) |
| **Tripo P1** | Premium | High (Low-poly) | High (PBR) | Yes | GLB, OBJ, FBX | High (Smart Mesh) | Moderate (Async) | No | Cloud | Moderate (Commercial) | Phase 1.5 (Multiview) |
| **TRELLIS.2** | Extreme | Extreme (Complex) | High (PBR) | No | GLB | High (O-Voxel) | High (Local/API) | Yes | 32GB+ VRAM | Low (MIT) | Phase 2 (OS Quality) |
| **Hunyuan3D 2.1** | High+ | High (Octree) | Extreme (PBR) | Yes | GLB, OBJ | Moderate | High (Local/API) | Yes | 24GB+ VRAM | Low (Apache 2.0) | Phase 3 (Server OS) |
| **SAM 3D** | Good | Moderate | Moderate | No | Mesh, GS | Low | High (Exp.) | Yes | TBD | Low (Meta) | Phase 4 (Objects) |
| **Apple SHARP** | Photoreal | N/A (GS Only) | N/A (GS) | No | GS (Splat) | None | Moderate (Local) | Yes | Apple Silicon | Low (Apple) | Phase 5 (Preview) |

## Provider Details

### Rodin / Hyper3D Gen-2
- **Best For**: High-quality production assets with clean quad topology.
- **Topology**: Advanced quad-meshing and rigging support.
- **PBR**: 4K texture support with full metallic/roughness maps.

### Meshy
- **Best For**: Versatile all-around 3D generation.
- **Workflow**: Two-stage (Preview -> Refine) process.
- **Topology**: Good quad/triangle options and polycount control.

### Tripo / Tripo P1
- **Best For**: Extremely fast, engine-ready low-poly assets.
- **Workflow**: Smart Mesh technology producing structured geometry in seconds.
- **Multiview**: Strong support for 2-4 input images.

### TRELLIS.2 (Future)
- **Status**: High-fidelity open-source candidate.
- **Strength**: Handles non-manifold and complex topologies via O-Voxel representation.

### Hunyuan3D 2.1 (Future)
- **Status**: Production-grade open-source candidate.
- **Strength**: High-fidelity texture synthesis (Paint) and DiT-based shape generation.

### SAM 3D Objects (Future)
- **Status**: Experimental object-aware reconstruction.
- **Strength**: Better handling of scene clutter and occlusions using SAM's "common sense".

### Apple SHARP (Future)
- **Status**: Spatial scene preview (Gaussian Splatting).
- **Strength**: Sub-second photorealistic view synthesis, though limited to nearby viewpoints.

## Strategy Summary
1. **SF3D**: Default self-hosted baseline for all users.
2. **Phase 1.5**: Integrate **Rodin** or **Meshy** as external premium benchmarks (opt-in).
3. **Phase 2+**: Evaluate server-side self-hosting for **TRELLIS.2** or **Hunyuan3D 2.1**.
