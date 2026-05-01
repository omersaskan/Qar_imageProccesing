"""
Reconstruction scorecard — single JSON document summarizing every quality
metric the pipeline computes for a given job.

Schema v1 (top-level keys):
    schema_version       : int
    job_id               : str
    generated_at         : ISO-8601 UTC
    coverage             : CoverageReport.to_dict()
    geometry             : GeometricReport.to_dict()
    texture              : TextureQualityAnalyzer.analyze_image() output (or None)
    reconstruction       : { engine, faces, vertices, has_texture, mesh_path, ... }
    capture_profile      : preset key + flags (from extraction_manifest)
    color_profile        : product/background RGB + category
    overall              : { grade: A|B|C|F, production_ready: bool,
                              review_required: bool, blockers: [...] }

The scorecard is **append-only** — written next to manifest.json on every
finalize.  Validator picks it up alongside (or instead of) the legacy
ValidationReport — see `validator.py` integration.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("scorecard")

SCHEMA_VERSION = 1


def _safe_load_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load {path}: {e}")
    return {}


def _grade_to_int(g: str) -> int:
    return {"A": 4, "B": 3, "C": 2, "F": 0}.get((g or "F").upper(), 0)


def _int_to_grade(n: int) -> str:
    return {4: "A", 3: "B", 2: "C", 0: "F", 1: "F"}.get(n, "F")


def build_scorecard(
    job_id: str,
    job_dir: Path,
    mesh: Optional[Any] = None,
    cameras: Optional[List[Any]] = None,
    masks: Optional[Dict[str, Any]] = None,
    point_cloud: Optional[Any] = None,
    texture_path: Optional[str] = None,
    expected_product_color: str = "unknown",
    extra_reconstruction_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a fresh scorecard document.  Caller writes it next to manifest.json.

    All inputs optional; missing inputs degrade individual sections to a
    safe default rather than failing the whole document.
    """
    from modules.qa_validation.coverage_metrics import compute_coverage_report
    from modules.qa_validation.geometric_quality import compute_geometric_report

    # Capture / color profile (from manifest if present)
    extraction_manifest = _safe_load_json(job_dir / "extraction_manifest.json")
    if not extraction_manifest:
        # Try one level up — captures/<id>/frames/extraction_manifest.json
        for parent in [job_dir, *list(job_dir.parents)[:3]]:
            for cand in parent.rglob("extraction_manifest.json"):
                extraction_manifest = _safe_load_json(cand)
                if extraction_manifest:
                    break
            if extraction_manifest:
                break
    color_profile = extraction_manifest.get("color_profile") or {}
    capture_profile = extraction_manifest.get("capture_profile") or {}
    capture_gate = extraction_manifest.get("capture_gate") or {}

    # 1. Coverage
    coverage = compute_coverage_report(
        cameras=cameras, mesh=mesh, masks=masks, point_cloud=point_cloud,
    )

    # 2. Geometry
    geometry = compute_geometric_report(mesh)

    # 3. Texture
    texture: Dict[str, Any] = {"status": "skipped", "reason": "no texture path"}
    if texture_path and Path(texture_path).exists():
        try:
            from modules.qa_validation.texture_quality import TextureQualityAnalyzer
            analyzer = TextureQualityAnalyzer()
            analyzed = analyzer.analyze_path(
                texture_path,
                expected_product_color=expected_product_color,
            )
            texture = analyzed
        except Exception as e:
            texture = {"status": "failed", "reason": str(e)}

    # 4. Reconstruction summary (from extra_reconstruction_fields + manifest)
    recon_manifest = _safe_load_json(job_dir / "manifest.json")
    reconstruction = {
        "engine_type": recon_manifest.get("engine_type"),
        "texturing_status": recon_manifest.get("texturing_status"),
        "vertex_count": (recon_manifest.get("mesh_metadata") or {}).get("vertex_count"),
        "face_count": (recon_manifest.get("mesh_metadata") or {}).get("face_count"),
        "has_texture": (recon_manifest.get("mesh_metadata") or {}).get("has_texture"),
        "uv_present": (recon_manifest.get("mesh_metadata") or {}).get("uv_present"),
        "checksum": recon_manifest.get("checksum"),
    }
    if extra_reconstruction_fields:
        reconstruction.update(extra_reconstruction_fields)

    # 5. Overall grade — conservative aggregation
    blockers: List[str] = []
    grades: List[int] = []

    # Geometry grade is the floor — broken mesh dominates everything
    grades.append(_grade_to_int(geometry.grade))
    if geometry.grade in ("F",):
        blockers.append(f"geometry: {', '.join(geometry.grade_reasons) or 'failed'}")

    # Coverage soft signals
    if coverage.observed_surface_ratio > 0:
        if coverage.observed_surface_ratio >= 0.70:
            grades.append(4)
        elif coverage.observed_surface_ratio >= 0.50:
            grades.append(3)
        elif coverage.observed_surface_ratio >= 0.30:
            grades.append(2)
        else:
            grades.append(0)
            blockers.append(f"observed_surface_ratio {coverage.observed_surface_ratio:.0%} <30%")
    if coverage.azimuth_coverage_ratio < 0.5 and coverage.sample_count > 0:
        blockers.append(f"azimuth_coverage {coverage.azimuth_coverage_ratio:.0%} <50%")
    if coverage.multi_height_score < 0.34 and coverage.sample_count > 0:
        blockers.append("multi_height_score <0.34 — only one elevation band captured")

    # Texture
    tex_status = (texture or {}).get("status")
    if tex_status == "pass":
        grades.append(4)
    elif tex_status == "review":
        grades.append(3)
    elif tex_status == "fail":
        grades.append(0)
        blockers.append(f"texture quality fail: {texture.get('reason', '?')}")

    # Reconstruction
    fc = reconstruction.get("face_count") or 0
    if reconstruction.get("has_texture") is True and reconstruction.get("uv_present") is True:
        grades.append(4)
    elif fc > 1000:
        grades.append(2)
    else:
        grades.append(0)
        blockers.append("reconstruction produced no usable mesh")

    # Sprint 2: factor capture_gate decision into overall grade
    gate_decision = (capture_gate or {}).get("decision")
    if gate_decision == "reshoot":
        grades.append(0)
        for r in (capture_gate or {}).get("reasons", []):
            blockers.append(f"capture_gate: {r}")
    elif gate_decision == "review":
        grades.append(2)

    overall_int = min(grades) if grades else 0
    overall_grade = _int_to_grade(overall_int)
    production_ready = overall_grade == "A" and not blockers
    review_required = overall_grade in ("B", "C") or bool(blockers)

    scorecard = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": coverage.to_dict(),
        "geometry": geometry.to_dict(),
        "texture": texture,
        "reconstruction": reconstruction,
        "capture_profile": capture_profile,
        "color_profile": color_profile,
        "capture_gate": capture_gate,
        "overall": {
            "grade": overall_grade,
            "production_ready": production_ready,
            "review_required": review_required,
            "blockers": blockers,
        },
    }
    return scorecard


def write_scorecard(job_dir: Path, scorecard: Dict[str, Any]) -> Path:
    """Atomic-ish write next to manifest.json.  Returns the written path."""
    job_dir.mkdir(parents=True, exist_ok=True)
    out_path = job_dir / "quality_report.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2, ensure_ascii=False)
    tmp_path.replace(out_path)
    return out_path
