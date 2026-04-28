# Evaluation Dataset Expansion Plan (SAM2/Segmentation)
**Date:** 2026-04-28
**Status:** PROPOSED

## 1. Objective
Expand the internal segmentation benchmark dataset to prevent "easy baseline" bias and ensure that future SAM2 (or other AI model) acceptance decisions are based on a representative sample of production challenges.

## 2. Dataset Requirements
- **Total Captures**: Minimum 5 new real-world captures.
- **Ground Truth (GT)**: Minimum 5 valid, manually audited GT frames per capture.
- **Diversity**: Each capture must represent one of the identified "hard cases".

## 3. Targeted Object Categories
To challenge the segmentation pipeline, the following object types must be included:

| Category | Challenge | Example |
| :--- | :--- | :--- |
| **Reflective** | Highlights/specularities causing mask fragmentation. | Chrome bottle, glossy ceramic. |
| **Transparent** | Foreground/background ambiguity. | Glass jar, plastic packaging. |
| **Low-Feature** | Lack of distinct edges or textures. | Matte white electronics, smooth egg. |
| **Food/Dessert** | Organic, complex textures and semi-transparency. | Cake with frosting, fruit bowl. |
| **Thin-Structure** | Fragile boundaries that fail in temporal propagation. | Plant stems, wire rack, chair legs. |

## 4. Acceptance Criteria for Expansion
1. **Audit Level**: All GT masks must be pixel-perfect (manual brush cleanup).
2. **Metadata**: Each capture must include a `metadata.json` identifying frame validity and specific challenges.
3. **Multi-View Coverage**: GT frames should be sampled from diverse angles (Low, Medium, High).

## 5. Decision Gate Policy
- **No Single-Capture Success**: SAM2 or any future segmentation method will NOT be promoted to production based on a single high-baseline capture.
- **Aggregate Performance**: Promotion requires an aggregate IoU Gain >= +0.05 across the *entire* expanded dataset.
- **Hard-Case Pass**: The model must show improved stability specifically in the "Thin-Structure" and "Reflective" categories before final sign-off.

---
> [!IMPORTANT]
> **Blocking Status**: Depth Anything development remains BLOCKED until this expanded dataset is processed and a model passes the aggregate threshold.
