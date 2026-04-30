# Meshysiz Product Asset Factory Docs

This directory contains the project documentation used by operators and implementation agents.

## Current source-of-truth docs for AI / WebGL phases

Use the following documents for all future Phase B through Phase G work:

1. [AI 3D Phases B-G Specification](AI_3D_PHASES_B_TO_G.md)
2. [AI 3D Safety Invariants](AI_3D_SAFETY_INVARIANTS.md)
3. [Phase F WebGL Review Experience](PHASE_F_WEBGL_REVIEW.md)
4. [Agent Implementation Plan](AI_3D_AGENT_IMPLEMENTATION_PLAN.md)

## Production principle

Photogrammetry remains the production source of truth.

AI, SAM3D, Meshy, and other generative 3D outputs are review-only diagnostic artifacts unless a future explicit production policy says otherwise.

AI-generated assets must never automatically:

- clear `recapture_required`
- bypass coverage gates
- bypass `quality_manifest` gates
- bypass validation gates
- set `publish_state=published`
- change the registry active pointer
- overwrite the photogrammetry manifest
- mark themselves as `production_pass`

## Legacy docs

Older docs in this folder may describe previous guided capture, dataset, reliability, or operator workflows. They remain useful as historical context, but for Phase B-G AI/WebGL work the documents listed above take precedence.
