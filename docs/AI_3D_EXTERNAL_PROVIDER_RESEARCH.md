# AI 3D External Provider Research

This document evaluates external and future internal providers for the AI 3D generation pipeline.

## Provider Comparison Matrix

| Provider | Expected Quality | Geometry Strength | Texture/PBR Strength | Multi-image Support | Output Formats | Topology Controls | API Complexity | Self-hosted Support | Hardware Needs | License/Privacy Risk | Recommended Phase |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SF3D** | High (Fast) | Good (Tri) | High (Unlit/PBR) | No | GLB | Low | Yes (Local/WSL) | 8GB VRAM (6.2GB Peak) | Stability AI Community (Revenue Threshold) | Phase 1 (Baseline) |
| **Rodin (Gen-2)** | Premium+ | Extreme (Quad) | Extreme (PBR) | Yes (1-5) | GLB, OBJ, FBX, USDZ, STL | High (Quad/Raw) | Moderate (Async) | No | Cloud | External Commercial / Privacy / Cost | Phase 1.5 (Premium) |
| **Meshy** | High | High (Quad) | High (PBR) | Yes (1-4) | GLB, OBJ, FBX, STL, USDZ, 3MF | High (Quad/Tri) | Moderate (Async) | No | Cloud | External Commercial / Privacy / Cost | Phase 1.5 (All-around) |
| **Tripo P1** | Premium | High (Low-poly) | High (PBR) | Yes | GLB, OBJ, FBX | High (Smart Mesh) | Moderate (Async) | No | Cloud | External Commercial / Privacy / Cost | Phase 1.5 (Multiview)* |
| **TRELLIS.2** | Extreme | Extreme (Complex) | High (PBR) | No | GLB | High (O-Voxel) | High (Local/API) | Yes | H100-class (Server) | To be verified | Future Server Provider Phase |
| **Hunyuan3D 2.1** | High+ | High (Octree) | Extreme (PBR) | Yes | GLB, OBJ | Moderate | High (Local/API) | Yes | 10-29GB VRAM (Texture Heavy) | Tencent Hunyuan Community (Territory Restr.) | Future Server PBR Provider Phase |
| **SAM 3D** | Good | Moderate | Moderate | No | Mesh, GS | Low | High (Exp.) | Yes | TBD | Low (Meta) | Future Object-Aware Provider Phase |
| **Apple SHARP** | Photoreal | N/A (GS Only) | N/A (GS) | No | GS (Splat) | None | Moderate (Local) | Yes | Apple Silicon | Low (Apple) | Future Spatial Preview Phase |

\* *Tripo P1 API details to be verified before implementation.*

## Provider Details

### SF3D (Stable Fast 3D)
- **Status**: Default self-hosted baseline.
- **License**: Stability AI Community License. Users must adhere to revenue thresholds and specific usage terms.
- **Performance**: High-speed local inference. Runs successfully on mobile workstation GPUs (e.g., RTX 5060 Laptop 8GB).

### Rodin / Hyper3D Gen-2
- **Best For**: High-quality production assets with clean quad topology.
- **Task Lifecycle**: Async (Create -> Poll -> Download).
- **Features**: 
    - Supports up to 5 reference images.
    - Material modes: PBR, Shaded, or All.
    - Mesh modes: Quad or Raw.
    - Quality overrides and "HighPack" 4K texture options.
- **Risks**: External data processing (privacy) and usage costs.

### Meshy (Meshy-6 / Latest)
- **Best For**: Versatile external benchmark.
- **Features**: 
    - Image-to-3D and Multi-image (1-4 images).
    - HD 4K base color texture support.
    - Specific topology control (Quad/Triangle) and target polycount.
    - Wide format support (GLB, OBJ, FBX, STL, USDZ, 3MF).

### Tripo / Tripo P1
- **Best For**: Rapid generation of structured low-poly meshes.
- **Technology**: Unified probabilistic spatial framework for clean topology.
- **Status**: Verify API parameters for parts segmentation and quad-remeshing before implementation.

### TRELLIS.2 (Future)
- **Model**: 4B parameter flow-matching transformer.
- **Representation**: O-Voxel (Field-free) allowing arbitrary topology.
- **Quality**: High-resolution generation (512³ to 1536³).
- **Hosting**: Requires server-grade hardware (H100/A100 class) for published benchmarks.

### Hunyuan3D 2.1 (Future)
- **License**: Tencent Hunyuan Community License. Restricted in certain territories (EU/UK/South Korea), flagged as `extra_gated_eu_disallowed`.
- **Hardware**: 
    - Shape generation: ~10GB VRAM.
    - Texture synthesis: ~21GB VRAM.
    - Combined pipeline: ~29GB VRAM.
- **Focus**: High-fidelity PBR texture synthesis.

### SAM 3D Objects (Future)
- **Focus**: Object-aware/clutter-aware 3D reconstruction from natural images.
- **Tech**: Generative foundation model for shape, texture, and layout.

### Apple SHARP (Future)
- **Focus**: Spatial/Gaussian scene preview.
- **Technology**: 3D Gaussian Splatting for sub-second view synthesis.
- **Output**: Not a GLB asset provider; primarily for volumetric rendering.
