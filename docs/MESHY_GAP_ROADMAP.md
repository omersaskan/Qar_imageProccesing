# Meshy Gap Roadmap: AI-Enhanced 3D Asset Factory

This roadmap outlines the transition from a purely deterministic photogrammetry pipeline (COLMAP/OpenMVS) to an AI-augmented system designed to solve the "Meshy Gaps"—the fundamental limitations of traditional reconstruction.

## 1. Capabilities of Deterministic Photogrammetry
The current COLMAP/OpenMVS pipeline is highly effective for:
*   **Geometric Fidelity**: Capturing sub-millimeter detail on high-feature surfaces.
*   **Metric Accuracy**: Providing real-world scale when paired with known camera intrinsics or reference objects.
*   **Texture Mapping**: Projecting high-resolution source imagery onto reconstructed geometry without "AI hallucinations."
*   **Static Scene Stability**: Robust reconstruction of rigid, non-moving objects in controlled lighting.

## 2. Fundamental Limitations (The "Gaps")
Deterministic methods reliably fail in the following scenarios:
*   **Invisible Surfaces**: Cannot reconstruct what is not seen (e.g., the "back" of an object in a 180-degree capture).
*   **Bottom Geometry**: Almost always missing due to contact with the support surface (table/floor).
*   **Low-Feature Objects**: Smooth, monochrome surfaces (e.g., a white plastic bottle) lack the gradients needed for feature matching.
*   **Reflective/Transparent Materials**: Specular highlights move relative to the camera, breaking the "brightness constancy" assumption of photogrammetry.
*   **Texture Completion**: Resulting textures have "black holes" where geometry was occluded or visibility was poor.
*   **Semantic Separation**: Hard to distinguish between the "product" and "support" based purely on geometry.
*   **Hole Filling and Topology Cleanup**: Raw outputs often contain non-manifold geometry, holes, and internal "floaters."

## 3. AI Enhancement Options

### A. SAM2 (Segment Anything Model v2)
*   **Integration Point**: Capture Workflow (Frame Extraction).
*   **Expected Quality Gain**: Perfect product/background separation; eliminates background noise from the sparse cloud.
*   **Compute Cost**: Moderate (GPU required for batch inference).
*   **Risk**: Mask flickering in video sequences (mitigated by SAM2 temporal consistency).
*   **Dependency**: PyTorch, NVIDIA GPU (8GB+ VRAM).
*   **MVP Path**: Run SAM2 on 10% of frames; interpolate masks.
*   **Production Path**: Full video segmentation with tracking.

### B. Depth Anything / Metric Depth Priors
*   **Integration Point**: Reconstruction Engine (Stereo Fusion).
*   **Expected Quality Gain**: Fills holes in low-feature areas by providing a "best guess" depth map.
*   **Compute Cost**: Low to Moderate (per-frame inference).
*   **Risk**: Scale mismatch between AI depth and COLMAP metric depth.
*   **Dependency**: Depth Anything V2 weights.
*   **MVP Path**: Use AI depth as a "voting" mechanism for COLMAP candidates.
*   **Production Path**: Monocular depth as a hard constraint for PatchMatchStereo.

### C. Multi-View Depth Consistency
*   **Integration Point**: Reconstruction Engine (Dense Reconstruction).
*   **Expected Quality Gain**: Drastically reduces noise in thin or semi-transparent structures.
*   **Compute Cost**: High.
*   **Risk**: Long processing times.
*   **Dependency**: Consistent depth estimation models.
*   **MVP Path**: Post-process COLMAP depth maps.
*   **Production Path**: End-to-end neural densification.

### D. Object-Centric Normal Estimation
*   **Integration Point**: Asset Cleanup (Alignment/Remeshing).
*   **Expected Quality Gain**: Smoother surfaces on monochrome objects; better edge definition.
*   **Compute Cost**: Low.
*   **Risk**: Over-smoothing of intentional high-frequency detail.
*   **Dependency**: Surface normal estimation models.
*   **MVP Path**: Use normals to guide Laplacian smoothing.
*   **Production Path**: Integrate normals into Poisson reconstruction weights.

### E. Texture Inpainting (Stable Diffusion / ControlNet)
*   **Integration Point**: Export Pipeline (Post-Texturing).
*   **Expected Quality Gain**: Removes shadows, fills occlusion gaps, and synthesizes "bottom" textures.
*   **Compute Cost**: High (requires heavy diffusion inference).
*   **Risk**: Visual "uncanny valley" or style mismatch with original photos.
*   **Dependency**: Stable Diffusion XL / ControlNet Inpaint.
*   **MVP Path**: Inpaint small holes only (<5% of texture area).
*   **Production Path**: Generative synthesis of unobserved surfaces (e.g., product bottom).

### F. Gaussian Splatting / NeRF Intermediate Representation
*   **Integration Point**: Reconstruction Engine (Intermediate Rep).
*   **Expected Quality Gain**: Superior handling of reflections and transparency; extremely fast visual feedback.
*   **Compute Cost**: High (Training) / Low (Rendering).
*   **Risk**: Difficult to convert back to standard GLB/Mesh topology without "fuzziness."
*   **Dependency**: 3DGS or Nerfstudio.
*   **MVP Path**: Use Splatting for rapid QA visualization only.
*   **Production Path**: Splat-to-Mesh conversion for final delivery.

### G. Generative Mesh Completion
*   **Integration Point**: Asset Cleanup (Hole Filling).
*   **Expected Quality Gain**: Synthesizes missing bottom/back geometry based on object class.
*   **Compute Cost**: High.
*   **Risk**: Geometric hallucinations (e.g., wrong type of legs for a chair).
*   **Dependency**: Large Reconstruction Models (LRMs) like TripoSR.
*   **MVP Path**: Apply only to "bottom" holes.
*   **Production Path**: Full 360 reconstruction from partial input.

## 4. AI Hallucination Safety Policy
To maintain product integrity, the following safety constraints apply to all AI enhancements:
*   **Synthesized Marking**: All AI-generated geometry or texture regions must be flagged in the asset metadata as "synthesized."
*   **Critical Region Protection**: Logos, labels, text, product-critical visual regions, and exact physical dimensions must NOT be subject to generative hallucination.
*   **Review Gating**: Generative completion (Phase 6.4/6.5) defaults to `review_ready` status. Assets are only marked `production_ready` after manual approval or high-confidence automated validation.

## 5. Phased Implementation Order

### Phase 6.0: Evaluation Dataset + Baseline Metrics
*   **Goal**: Establish a ground-truth dataset to quantify AI gains.
*   **Components**:
    *   Curate 5–10 real video datasets representing "Meshy Gaps" (reflective, low-feature, missing bottom).
    *   Generate manually labeled masks for selected validation frames.
    *   Calculate baseline dense mask leakage metrics and reconstruction success rates.
    *   Establish a standardized before/after comparison protocol for all future AI modules.

### Phase 6.1: Capture QA & Segmentation (SAM2)
*   **Goal**: Hardened input gating followed by semantic segmentation upgrade.
*   **Order**: Capture QA hardening runs first or in parallel with SAM2.
*   **Acceptance Criteria**: 
    *   **Capture QA**: Bad videos fail early before downstream processing.
    *   **Segmentation**: Mask IoU >= 0.92 on labeled validation frames; background leakage ratio <= 2%.

### Phase 6.2: Depth & Normal Priors
*   **Goal**: Use monocular depth (Depth Anything) and surface normals to guide COLMAP through low-feature regions.
*   **Acceptance Criteria**: Hole area reduced by >= 50% compared to deterministic baseline on low-feature objects.

### Phase 6.3: Texture Inpainting
*   **Goal**: Use Image-to-Image diffusion to fill UV gaps and remove lighting artifacts.
*   **Acceptance Criteria**: Empty/black texture ratio reduced to <= 1–3% of total atlas area.

### Phase 6.4: Optional Gaussian/NeRF/Generative Completion
*   **Goal**: Volumetric representation for complex materials or full generative completion for unobserved surfaces.
*   **Acceptance Criteria**: Reflective objects become `review_ready` instead of failing; synthesized regions pass visual inspection.

---

## 🛑 User Review Required
**Implementation of any AI module in Phase 6 MUST wait for explicit approval of this roadmap.**

Please review the revised phases and safety policies. Once confirmed, we will begin Phase 6.0.
