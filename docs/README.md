# Meshysiz Product Asset Factory — Docs Index

This `docs/` directory is the source of truth for planned Phase B through Phase G work.

The current production architecture must remain photogrammetry-first. AI, SAM3D, Meshy, or any generative 3D output is allowed only as review-only, diagnostic, comparison, or manually reviewed non-production output until an explicit production policy says otherwise.

## Read these first

1. [`AI_PHASES_B_TO_G.md`](./AI_PHASES_B_TO_G.md) — full phase specification from Phase B to Phase G.
2. [`AI_AGENT_IMPLEMENTATION_PLAN.md`](./AI_AGENT_IMPLEMENTATION_PLAN.md) — PR-by-PR execution order for coding agents.
3. [`AI_PRODUCTION_GUARDRAILS.md`](./AI_PRODUCTION_GUARDRAILS.md) — hard safety rules and forbidden mutations.

## Non-negotiable production rule

Photogrammetry remains the production source of truth.

AI preview output must never automatically:

- clear `recapture_required`
- bypass coverage gates
- bypass `quality_manifest` gates
- bypass validation gates
- set `publish_state=published`
- update the registry active pointer
- overwrite a photogrammetry manifest
- become `production_pass`

## Current implementation posture

Agents should implement Phase B first as a safe no-op AI proxy shell. Do not implement real Meshy or SAM3D calls until the no-op provider, metadata contract, policy guards, and safety tests are passing.
