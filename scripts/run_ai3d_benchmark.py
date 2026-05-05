"""
Phase 3A — SF3D Local Benchmark Runner.

Runs generate_ai_3d() directly against the local SF3D provider across a matrix
of quality modes and background-removal settings.

Usage:
    py scripts/run_ai3d_benchmark.py \\
        --modes balanced,high,ultra \\
        --bg-modes off,on \\
        --limit 2

Flags:
    --allow-unavailable   Allow writing a report even if SF3D is disabled.
                          Report will be stamped "DRY RUN / PROVIDER UNAVAILABLE".
"""
import os
import sys
import json
import csv
import uuid
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# Add repo root to sys.path so imports work when run from any CWD
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from modules.ai_3d_generation.pipeline import generate_ai_3d
from modules.operations.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ai3d_benchmark")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()[:12]
    except Exception:
        return "unknown"


def _make_bench_id(stem: str, mode: str, bg: bool) -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    uid = uuid.uuid4().hex[:6]
    bg_tag = "bgon" if bg else "bgoff"
    return f"bench_{ts}_{uid}_{stem}_{mode}_{bg_tag}"


def _check_sf3d_enabled() -> tuple[bool, str]:
    """Return (available, reason) for the local SF3D provider."""
    if not getattr(settings, "ai_3d_generation_enabled", False):
        return False, "AI_3D_GENERATION_ENABLED is false"
    if not getattr(settings, "sf3d_enabled", False):
        return False, "SF3D_ENABLED is false"
    mode = getattr(settings, "sf3d_execution_mode", "disabled")
    if mode == "disabled":
        return False, "SF3D_EXECUTION_MODE is disabled"
    return True, "ok"


def _env_hint() -> str:
    return (
        "\nTo enable SF3D for a real benchmark, set these environment variables:\n"
        "  $env:AI_3D_GENERATION_ENABLED='true'\n"
        "  $env:SF3D_ENABLED='true'\n"
        "  $env:SF3D_EXECUTION_MODE='wsl_subprocess'\n"
        "  $env:AI_3D_BACKGROUND_REMOVAL_ENABLED='true'"
    )


def get_mesh_stats(glb_path) -> dict:
    stats = {
        "vertex_count": 0,
        "face_count": 0,
        "geometry_count": 0,
        "mesh_stats_available": False,
    }
    if not glb_path or not os.path.exists(glb_path):
        return stats
    try:
        import trimesh

        scene = trimesh.load(glb_path, force="scene")
        if isinstance(scene, trimesh.Scene):
            stats["geometry_count"] = len(scene.geometry)
            for mesh in scene.geometry.values():
                if hasattr(mesh, "vertices"):
                    stats["vertex_count"] += len(mesh.vertices)
                if hasattr(mesh, "faces"):
                    stats["face_count"] += len(mesh.faces)
        else:
            stats["geometry_count"] = 1
            stats["vertex_count"] = len(scene.vertices)
            stats["face_count"] = len(scene.faces)
        stats["mesh_stats_available"] = True
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("Failed to extract mesh stats from %s: %s", glb_path, exc)
    return stats


def _is_successful(row: dict) -> bool:
    """A run counts as successful only if SF3D produced a real GLB."""
    return (
        row.get("status") in ("ok", "review")
        and row.get("provider_status") == "ok"
        and bool(row.get("output_glb_path"))
        and os.path.exists(row.get("output_glb_path", ""))
    )


# ─── main ─────────────────────────────────────────────────────────────────────

def run_benchmark():
    parser = argparse.ArgumentParser(description="SF3D Local Benchmark Runner")
    parser.add_argument("--input-dir", default="scratch/ai3d_benchmark_inputs")
    parser.add_argument("--output-dir", default="reports/ai3d_benchmark")
    parser.add_argument("--modes", default="balanced,high,ultra")
    parser.add_argument("--bg-modes", default="off,on")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Write report even if SF3D is disabled (marks report as DRY RUN).",
    )
    args = parser.parse_args()

    # ── 1. Provider preflight ─────────────────────────────────────────────────
    # Force settings overrides so benchmark env works without shell exports
    settings.ai_3d_generation_enabled = True

    sf3d_ok, sf3d_reason = _check_sf3d_enabled()
    if not sf3d_ok:
        msg = f"SF3D unavailable: {sf3d_reason}."
        if not args.allow_unavailable:
            logger.error(msg)
            logger.error("Refusing to write benchmark report with zero successful runs.")
            logger.error(_env_hint())
            sys.exit(1)
        else:
            logger.warning(msg + " Proceeding with --allow-unavailable (DRY RUN).")

    dry_run = not sf3d_ok  # True when provider is unavailable

    # ── 2. Collect inputs ─────────────────────────────────────────────────────
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = args.modes.split(",")
    bg_options = [m.lower() == "on" for m in args.bg_modes.split(",")]

    inputs = sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if args.limit:
        inputs = inputs[: args.limit]

    logger.info(
        "Starting benchmark: %d inputs × %d modes × %d bg-modes",
        len(inputs), len(modes), len(bg_options),
    )

    sha = _git_sha()
    results = []

    for input_file in inputs:
        for mode in modes:
            for bg_removal in bg_options:
                bench_id = _make_bench_id(input_file.stem, mode, bg_removal)
                logger.info("--- %s ---", bench_id)

                try:
                    session_id = bench_id
                    session_dir = output_dir / session_id
                    session_dir.mkdir(parents=True, exist_ok=True)

                    manifest = generate_ai_3d(
                        session_id=session_id,
                        input_file_path=str(input_file),
                        output_base_dir=str(session_dir),
                        provider_name="sf3d",
                        options={
                            "quality_mode": mode,
                            "background_removal_enabled": bg_removal,
                        },
                    )

                    status = manifest.get("status")
                    glb_path = manifest.get("output_glb_path")
                    glb_size = manifest.get("output_size_bytes") or 0
                    peak_mem = manifest.get("peak_mem_mb") or 0
                    duration = manifest.get("duration_sec", 0)
                    pre = manifest.get("preprocessing", {})
                    mesh_stats = get_mesh_stats(glb_path)

                    ranking = manifest.get("candidate_ranking") or []
                    top_score = ranking[0].get("score") if ranking else None

                    ar = manifest.get("ar_readiness") or {}
                    row = {
                        "benchmark_id": bench_id,
                        "commit_sha": sha,
                        "input_filename": input_file.name,
                        "quality_mode": mode,
                        "background_removal_enabled": bg_removal,
                        "session_id": manifest.get("session_id"),
                        "status": status,
                        "provider_status": manifest.get("provider_status"),
                        "input_mode": manifest.get("input_mode"),
                        "candidate_count": manifest.get("candidate_count"),
                        "selected_candidate_id": manifest.get("selected_candidate_id"),
                        "duration_sec": duration,
                        "output_size_bytes": glb_size,
                        "peak_mem_mb": peak_mem,
                        "device": (manifest.get("worker_metadata") or {}).get("device"),
                        "input_size": (manifest.get("resolved_quality") or {}).get("input_size"),
                        "bg_removed": pre.get("background_removed"),
                        "mask_source": pre.get("mask_source"),
                        "foreground_ratio": pre.get("foreground_ratio_estimate"),
                        "score": top_score,
                        "vertex_count": mesh_stats["vertex_count"],
                        "face_count": mesh_stats["face_count"],
                        "mesh_stats_available": mesh_stats["mesh_stats_available"],
                        "output_glb_path": glb_path,
                        "prepared_image_path": manifest.get("prepared_image_path"),
                        "warnings_count": len(manifest.get("warnings") or []),
                        "errors_count": len(manifest.get("errors") or []),
                        "ar_score": ar.get("score"),
                        "ar_verdict": ar.get("verdict"),
                        "ar_warnings_count": len(ar.get("warnings") or []),
                    }
                    results.append(row)
                    logger.info(
                        "Result: %s (provider=%s) in %.1fs, GLB: %d bytes",
                        status, row["provider_status"], duration, glb_size,
                    )

                except Exception as exc:
                    logger.error("Benchmark failed for %s: %s", bench_id, exc)
                    results.append({
                        "benchmark_id": bench_id,
                        "commit_sha": sha,
                        "input_filename": input_file.name,
                        "quality_mode": mode,
                        "background_removal_enabled": bg_removal,
                        "status": "failed",
                        "provider_status": "failed",
                        "error": str(exc),
                    })

    # ── 3. Write outputs ──────────────────────────────────────────────────────
    success_count = sum(1 for r in results if _is_successful(r))

    if success_count == 0 and not args.allow_unavailable:
        logger.error(
            "0 successful runs produced. Not overwriting benchmark reports."
        )
        logger.error(_env_hint())
        sys.exit(1)

    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    csv_path = output_dir / "results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    md_path = output_dir / "AI_3D_PHASE3A_SF3D_BENCHMARK_REPORT.md"
    dry_label = " [DRY RUN / PROVIDER UNAVAILABLE]" if dry_run else ""
    phase_status = (
        "Benchmark attempted but no successful SF3D runs were recorded."
        if success_count == 0
        else f"{success_count} successful SF3D run(s) recorded."
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# AI 3D Phase 3A \u2014 SF3D Local Benchmark Report{dry_label}\n\n")
        f.write(f"- **Date**: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"- **Commit SHA**: `{sha}`\n")
        f.write(f"- **Environment**: Local Windows/WSL2\n")
        f.write(f"- **SF3D Available**: {'Yes' if sf3d_ok else 'No — ' + sf3d_reason}\n")
        f.write(f"- **Total Inputs**: {len(inputs)}\n")
        f.write(f"- **Total Runs**: {len(results)}\n")
        f.write(f"- **Successful Runs**: {success_count}\n")
        f.write(f"- **Phase Status**: {phase_status}\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Input | Mode | BG | Status | Provider | Duration | GLB Size | Peak VRAM | Score | AR Score | AR Verdict |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            if "error" in r and "status" not in r:
                f.write(f"| {r.get('input_filename','-')} | {r.get('quality_mode','-')} | - | EXCEPTION | - | - | - | - | - | - | - |\n")
                continue
            bg_str = "ON" if r.get("background_removal_enabled") else "OFF"
            size_bytes = r.get("output_size_bytes") or 0
            size_mb = round(size_bytes / (1024 * 1024), 2)
            f.write(
                f"| {r['input_filename']} | {r['quality_mode']} | {bg_str} "
                f"| {r['status']} | {r.get('provider_status','-')} "
                f"| {r['duration_sec']}s | {size_mb} MB "
                f"| {r.get('peak_mem_mb', 0)} MB | {r.get('score')} "
                f"| {r.get('ar_score', '-')} | {r.get('ar_verdict', '-')} |\n"
            )

        f.write("\n## Notes\n\n")
        f.write("- This benchmark covers only local SF3D.\n")
        f.write("- External providers remain disabled and were not touched.\n")
        f.write("- This is not true multi-view reconstruction.\n")
        f.write("- Mesh statistics collected via `trimesh` when available.\n")

    logger.info("Benchmark complete. Reports saved to %s", output_dir)
    logger.info("Successful runs: %d / %d", success_count, len(results))

    if success_count == 0:
        logger.warning("Phase 3A is NOT closed: 0 successful SF3D runs.")


if __name__ == "__main__":
    run_benchmark()
