import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add root to sys.path
ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT))

from modules.operations.settings import settings
from modules.utils.file_persistence import atomic_write_json

# Thresholds
THR_PROD_ACCEPTED_FRAMES = 30
THR_REVIEW_ACCEPTED_FRAMES = 15
THR_PROD_DENSE_POINTS = 25000
THR_REVIEW_DENSE_POINTS = 10000
THR_PROD_FALLBACK_WHITE = 0.05
THR_REVIEW_FALLBACK_WHITE = 0.10
THR_PROD_REG_RATIO = 0.70
THR_PROD_REG_COUNT = 20
THR_PROD_ISOLATION_CONF = 0.75
THR_REVIEW_ISOLATION_CONF = 0.60

def collect_evidence(job_id: str, workspace_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evidence = {
        "job_id": job_id,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "reports": {},
        "logs": [],
        "artifacts": {}
    }
    
    # 1. Config Snapshot
    evidence["config_snapshot"] = {
        "env": settings.env.value if hasattr(settings.env, "value") else str(settings.env),
        "recon_pipeline": settings.recon_pipeline,
        "require_textured_output": settings.require_textured_output,
        "recon_mesh_budget_faces": settings.recon_mesh_budget_faces,
        "recon_mobile_target_faces": settings.recon_mobile_target_faces,
    }
    
    # 2. Environment Snapshot
    evidence["environment_snapshot"] = {
        "os": os.name,
        "cwd": os.getcwd(),
        "python_version": sys.version,
    }

    # Locate job.json
    job_json_path = workspace_path / "job.json"
    job_data = {}
    if not job_json_path.exists():
        print(f"Warning: job.json not found at {job_json_path}")
    else:
        with open(job_json_path, "r") as f:
            job_data = json.load(f)
            evidence["reports"]["job_config"] = job_data

    session_id = job_data.get("capture_session_id") or job_data.get("session_id")
    
    # Locate reports (check both session and job based paths)
    data_root = Path(settings.data_root)
    reports_dirs = []
    if session_id:
        reports_dirs.append(data_root / "captures" / session_id / "reports")
    reports_dirs.append(workspace_path / "reports")
    
    report_mapping = {
        "capture_report": "coverage_report.json",
        "extraction_report": "quality_report.json",
        "export_metrics": "export_metrics.json",
        "validation_report": "validation_report.json",
        "guidance_report": "guidance_report.json"
    }
    
    for reports_dir in reports_dirs:
        if reports_dir.exists():
            for key, filename in report_mapping.items():
                if key in evidence["reports"]: continue # Don't overwrite if found in prioritized path
                path = reports_dir / filename
                if path.exists():
                    with open(path, "r") as f:
                        evidence["reports"][key] = json.load(f)
                    
    # Reconstruction Audit
    audit_path = workspace_path / "reconstruction_audit.json"
    if audit_path.exists():
        with open(audit_path, "r") as f:
            evidence["reports"]["reconstruction_audit"] = json.load(f)
            
    # Cleanup stats (Check session_id then job_id)
    cleanup_search_ids = []
    if session_id: cleanup_search_ids.append(session_id)
    cleanup_search_ids.append(job_id)
    cleanup_search_ids.append(f"job_{job_id}")
    
    for search_id in cleanup_search_ids:
        cleaned_dir = data_root / "cleaned" / search_id
        if cleaned_dir.exists():
            stats_path = cleaned_dir / "cleanup_stats.json"
            if stats_path.exists() and "cleanup_stats" not in evidence["reports"]:
                with open(stats_path, "r") as f:
                    evidence["reports"]["cleanup_stats"] = json.load(f)
            
            meta_path = cleaned_dir / "normalized_metadata.json"
            if meta_path.exists() and "normalized_metadata" not in evidence["reports"]:
                with open(meta_path, "r") as f:
                    evidence["reports"]["normalized_metadata"] = json.load(f)

    # Fusion Ablation Report
    ablation_path = workspace_path / "ablation_report.json"
    if ablation_path.exists():
        with open(ablation_path, "r") as f:
            evidence["reports"]["fusion_ablation"] = json.load(f)

    # Collect Logs
    log_files = list(workspace_path.glob("*.log"))
    for attempt_dir in workspace_path.glob("attempt_*"):
        log_files.extend(list(attempt_dir.glob("*.log")))
        
    for log_path in log_files:
        try:
            rel_path = log_path.relative_to(workspace_path)
        except ValueError:
            rel_path = Path(log_path.name)
        dest = output_dir / "logs" / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_path, dest)
        evidence["logs"].append(str(rel_path))

    # Artifact Tree
    tree_lines = []
    def build_tree(path: Path, indent: str = ""):
        tree_lines.append(f"{indent}{path.name}/" if path.is_dir() else f"{indent}{path.name}")
        if path.is_dir():
            try:
                for child in sorted(path.iterdir()):
                    if child.name in [".git", "__pycache__", "venv", ".pytest_cache"]:
                        continue
                    build_tree(child, indent + "  ")
            except PermissionError:
                tree_lines.append(f"{indent}  [Permission Denied]")
            except Exception as e:
                tree_lines.append(f"{indent}  [Error: {e}]")
    
    build_tree(workspace_path)
    with open(output_dir / "artifact_tree.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(tree_lines))

    # Final QA Decision Checklist
    checklist = generate_checklist(evidence)
    evidence["delivery_checklist"] = checklist

    # Save reports
    atomic_write_json(output_dir / "evidence_report.json", evidence)
    
    # Generate Markdown
    generate_markdown_report(evidence, output_dir / "evidence_report.md")
    
    print(f"Evidence bundle exported to {output_dir}")
    print(f"Final Status: {checklist['final_status']}")

def generate_checklist(evidence: Dict[str, Any]) -> Dict[str, Any]:
    reports = evidence.get("reports", {})
    cap_report = reports.get("capture_report", {})
    ext_report = reports.get("extraction_report", {})
    val_report = reports.get("validation_report", {})
    audit = reports.get("reconstruction_audit", {})
    cleanup_stats = reports.get("cleanup_stats", {})
    export_metrics = reports.get("export_metrics", {})
    config = evidence.get("config_snapshot", {})
    require_textured = config.get("require_textured_output", False)
    
    best_attempt = None
    if audit and audit.get("attempts") and audit.get("selected_best_index") is not None:
        best_attempt = audit["attempts"][audit["selected_best_index"]]
    
    metadata = best_attempt.get("metadata", {}) if best_attempt else {}
    
    # 1. Data Normalization
    dense_image_count = (
        metadata.get("dense_image_count") or 
        metadata.get("dense_images_count") or 
        (best_attempt.get("dense_image_count") if best_attempt else 0) or 
        (best_attempt.get("registered_images") if best_attempt else 0) or 0
    )
    
    dense_mask_count = metadata.get("dense_mask_count") or metadata.get("dense_mask_exact_matches") or metadata.get("dense_mask_exact_filename_matches") or 0
    dense_mask_exact = metadata.get("dense_mask_exact_filename_matches") or metadata.get("dense_mask_exact_matches") or 0
    dense_mask_dim = metadata.get("dense_mask_dimension_matches") or 0
    
    accepted_frames = ext_report.get("saved_count", 0)
    registered_images = best_attempt.get("registered_images", 0) if best_attempt else 0
    reg_ratio = registered_images / accepted_frames if accepted_frames > 0 else 0.0
    
    points = best_attempt.get("dense_points_fused", 0) if best_attempt else 0
    fallback_ratio = metadata.get("dense_mask_fallback_white_ratio", 1.0)
    
    # Support both nested and flat structure
    isolation_method = cleanup_stats.get("isolation", {}).get("object_isolation_method") or cleanup_stats.get("object_isolation_method")
    isolation_conf = cleanup_stats.get("isolation", {}).get("isolation_confidence", 0.0) or cleanup_stats.get("isolation_confidence", 0.0)
    
    failure_reasons = []
    warning_reasons = []
    
    items = {}
    
    def check_item(key: str, prod_cond: bool, review_cond: bool, value: Any, required: str, fail_msg: str, warn_msg: str):
        if prod_cond:
            status = "production_ready"
        elif review_cond:
            status = "review_ready"
            warning_reasons.append(warn_msg)
        else:
            status = "failed"
            failure_reasons.append(fail_msg)
        
        items[key] = {
            "status": status,
            "value": value,
            "required": required
        }

    # 2. Evaluate Items
    
    # Capture Status
    cap_val = cap_report.get("overall_status")
    check_item(
        "capture_status",
        cap_val in ["sufficient", "PASS"],
        cap_val in ["warn", "REVIEW"],
        cap_val,
        "PROD: PASS/sufficient, REVIEW: warn",
        f"Capture status is {cap_val}",
        f"Capture status is {cap_val} (Review required)"
    )

    # Accepted Frames
    check_item(
        "accepted_frames",
        accepted_frames >= THR_PROD_ACCEPTED_FRAMES,
        accepted_frames >= THR_REVIEW_ACCEPTED_FRAMES,
        accepted_frames,
        f"PROD: >= {THR_PROD_ACCEPTED_FRAMES}, REVIEW: >= {THR_REVIEW_ACCEPTED_FRAMES}",
        f"Too few frames: {accepted_frames}",
        f"Low frame count: {accepted_frames}"
    )

    # Dense Masks
    mask_match_prod = (dense_mask_exact == dense_image_count and dense_mask_dim == dense_image_count and dense_image_count > 0)
    # User requirement: fail when dense mask counts are nonzero but not equal to dense_image_count
    # This implies review is not allowed for mismatches.
    mask_match_review = mask_match_prod 
    check_item(
        "dense_masks_integrity",
        mask_match_prod,
        mask_match_review,
        f"exact={dense_mask_exact}, dim={dense_mask_dim}, total={dense_image_count}",
        "PROD/REVIEW: exact==total and dim==total",
        f"Dense mask mismatch or missing (exact={dense_mask_exact}/{dense_image_count})",
        "N/A"
    )

    # Fallback White Ratio
    check_item(
        "fallback_white_ratio",
        fallback_ratio <= THR_PROD_FALLBACK_WHITE,
        fallback_ratio <= THR_REVIEW_FALLBACK_WHITE,
        f"{fallback_ratio:.3f}",
        f"PROD: <= {THR_PROD_FALLBACK_WHITE}, REVIEW: <= {THR_REVIEW_FALLBACK_WHITE}",
        f"Excessive mask fallback: {fallback_ratio:.3f}",
        f"High mask fallback: {fallback_ratio:.3f}"
    )

    # Registered Images & Ratio
    reg_prod = (registered_images >= THR_PROD_REG_COUNT and reg_ratio >= THR_PROD_REG_RATIO)
    reg_review = (registered_images >= 10)
    check_item(
        "registered_images",
        reg_prod,
        reg_review,
        f"count={registered_images}, ratio={reg_ratio:.2f}",
        f"PROD: >= {THR_PROD_REG_COUNT} & {THR_PROD_REG_RATIO:.0%}",
        f"Low registration: {registered_images} ({reg_ratio:.2%})",
        f"Moderate registration: {registered_images} ({reg_ratio:.2%})"
    )

    # Fused Points
    check_item(
        "fused_point_count",
        points >= THR_PROD_DENSE_POINTS,
        points >= THR_REVIEW_DENSE_POINTS,
        points,
        f"PROD: >= {THR_PROD_DENSE_POINTS}, REVIEW: >= {THR_REVIEW_DENSE_POINTS}",
        f"Insufficient density: {points} pts",
        f"Moderate density: {points} pts"
    )

    # Isolation Method
    check_item(
        "object_isolation_method",
        isolation_method in ["mask_guided", "hybrid_pc_mask"],
        isolation_method == "geometric_only",
        isolation_method,
        "PROD: mask_guided/hybrid",
        f"Unsupported isolation: {isolation_method}",
        "Geometric-only isolation (Verify background removal)"
    )

    # Isolation Confidence
    check_item(
        "isolation_confidence",
        isolation_conf >= THR_PROD_ISOLATION_CONF,
        isolation_conf >= THR_REVIEW_ISOLATION_CONF,
        f"{isolation_conf:.2f}",
        f"PROD: >= {THR_PROD_ISOLATION_CONF}, REVIEW: >= {THR_REVIEW_ISOLATION_CONF}",
        f"Low isolation confidence: {isolation_conf:.2f}",
        f"Suboptimal isolation confidence: {isolation_conf:.2f}"
    )

    # Texture Components
    tex_count = export_metrics.get("texture_count", 0)
    mat_count = export_metrics.get("material_count", 0)
    tex_uv = export_metrics.get("all_textured_primitives_have_texcoord_0", False)
    tex_app = export_metrics.get("texture_applied", False)
    
    texture_gate_prod = (tex_count > 0 and mat_count > 0 and tex_uv and tex_app)
    texture_gate_review = texture_gate_prod
    
    if require_textured:
        if not texture_gate_prod:
            failure_reasons.append("Config requires texture but one or more texture components are missing.")
    
    check_item("texture_count", tex_count > 0, tex_count > 0, tex_count, "> 0", "No textures found", "N/A")
    check_item("material_count", mat_count > 0, mat_count > 0, mat_count, "> 0", "No materials found", "N/A")
    check_item("texcoord_0_exists", tex_uv, tex_uv, tex_uv, "True", "Missing UV coords", "N/A")
    check_item("texture_applied", tex_app, tex_app, tex_app, "True", "Texture not applied to mesh", "N/A")

    # GLB Validation (Fallback to export_metrics if report missing)
    glb_val = val_report.get("final_decision")
    if not glb_val and export_metrics:
        if export_metrics.get("delivery_ready") is True:
            glb_val = "pass"
        elif export_metrics.get("export_status") == "success":
            glb_val = "review"
    
    check_item(
        "glb_validation",
        glb_val == "pass",
        glb_val == "review",
        glb_val,
        "PROD: pass, REVIEW: review",
        f"GLB validation failed or missing: {glb_val}",
        f"GLB requires manual review: {glb_val}"
    )

    # 3. Final Decision
    final_status = "production_ready"
    for item in items.values():
        if item["status"] == "failed":
            final_status = "failed"
            break
        elif item["status"] == "review_ready":
            final_status = "review_ready"
            # Keep checking in case a 'failed' exists later
    
    # Add extra fields to checklist for report
    items["final_status"] = final_status
    items["failure_reasons"] = failure_reasons
    items["warning_reasons"] = warning_reasons
    items["dense_image_count"] = dense_image_count
    items["dense_mask_count"] = dense_mask_count
    items["dense_mask_exact_matches"] = dense_mask_exact
    items["dense_mask_dimension_matches"] = dense_mask_dim
    items["dense_mask_fallback_white_ratio"] = fallback_ratio
    items["registered_image_ratio"] = reg_ratio
    items["mask_support_ratio"] = cleanup_stats.get("isolation", {}).get("mask_support_ratio")
    items["point_cloud_support_ratio"] = cleanup_stats.get("isolation", {}).get("point_cloud_support_ratio")

    return items

def generate_markdown_report(evidence: Dict[str, Any], output_path: Path):
    job_id = evidence["job_id"]
    checklist = evidence.get("delivery_checklist", {})
    final_status = checklist.get("final_status", "failed")
    
    status_colors = {
        "production_ready": "✅ PRODUCTION READY",
        "review_ready": "⚠️ REVIEW READY",
        "failed": "❌ FAILED"
    }
    
    lines = [
        f"# Reconstruction Evidence Report: {job_id}",
        f"**Status:** {status_colors.get(final_status, final_status)}",
        f"**Exported At:** {evidence['exported_at']}",
        "",
        "## Delivery Checklist Summary",
        "| Criterion | Status | Value | Required |",
        "|-----------|--------|-------|----------|",
    ]
    
    # Filter out non-item fields for the table
    item_keys = [k for k in checklist.keys() if isinstance(checklist[k], dict) and "status" in checklist[k]]
    
    for key in item_keys:
        item = checklist[key]
        status_map = {
            "production_ready": "✅",
            "review_ready": "⚠️",
            "failed": "❌"
        }
        status_icon = status_map.get(item["status"], "❓")
        lines.append(f"| {key.replace('_', ' ').title()} | {status_icon} | {item['value']} | {item['required']} |")
    
    if checklist.get("failure_reasons"):
        lines.append("")
        lines.append("### ❌ Failure Reasons")
        for reason in checklist["failure_reasons"]:
            lines.append(f"- {reason}")
            
    if checklist.get("warning_reasons"):
        lines.append("")
        lines.append("### ⚠️ Warning Reasons")
        for reason in checklist["warning_reasons"]:
            lines.append(f"- {reason}")
            
    lines.append("")
    if final_status == "production_ready":
        lines.append("> [!TIP]")
        lines.append("> Asset is PRODUCTION READY. Automated checks passed with high confidence.")
    elif final_status == "review_ready":
        lines.append("> [!IMPORTANT]")
        lines.append("> Asset is REVIEW READY. Manual inspection of geometric quality and background removal is required.")
    else:
        lines.append("> [!CAUTION]")
        lines.append("> Asset FAILED delivery criteria. Do not deliver without remediation.")
        
    lines.append("")
    lines.append("## Configuration Snapshot")
    lines.append("```json")
    lines.append(json.dumps(evidence["config_snapshot"], indent=2))
    lines.append("```")
    
    lines.append("")
    lines.append("## Detailed Metrics")
    metrics_fields = [
        "dense_image_count", "dense_mask_count", "dense_mask_exact_matches", 
        "dense_mask_dimension_matches", "dense_mask_fallback_white_ratio",
        "registered_image_ratio", "mask_support_ratio", "point_cloud_support_ratio"
    ]
    for field in metrics_fields:
        val = checklist.get(field)
        if val is not None:
            lines.append(f"- **{field.replace('_', ' ').title()}:** {val}")

    lines.append("")
    lines.append("## Available Logs")
    for log in evidence["logs"]:
        lines.append(f"- {log}")
        
    lines.append("")
    lines.append("## Artifacts")
    lines.append("See `artifact_tree.txt` for full hierarchy.")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    parser = argparse.ArgumentParser(description="Export Reconstruction Evidence Bundle")
    parser.add_argument("--job-id", required=True, help="Job ID")
    parser.add_argument("--workspace", required=True, help="Path to job workspace")
    parser.add_argument("--output-dir", required=True, help="Path to export bundle")
    
    args = parser.parse_args()
    collect_evidence(args.job_id, Path(args.workspace), Path(args.output_dir))

if __name__ == "__main__":
    main()
