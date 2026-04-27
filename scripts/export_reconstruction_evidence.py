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

def collect_evidence(job_id: str, workspace_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evidence = {
        "job_id": job_id,
        "exported_at": datetime.utcnow().isoformat(),
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
    if not job_json_path.exists():
        print(f"Warning: job.json not found at {job_json_path}")
        job_data = {}
    else:
        with open(job_json_path, "r") as f:
            job_data = json.load(f)
            evidence["reports"]["job_config"] = job_data

    session_id = job_data.get("capture_session_id")
    
    # Locate reports
    data_root = Path(settings.data_root)
    reports_dir = data_root / "captures" / session_id / "reports" if session_id else None
    
    report_mapping = {
        "capture_report": "coverage_report.json",
        "extraction_report": "quality_report.json",
        "export_metrics": "export_metrics.json",
        "validation_report": "validation_report.json",
        "guidance_report": "guidance_report.json"
    }
    
    if reports_dir and reports_dir.exists():
        for key, filename in report_mapping.items():
            path = reports_dir / filename
            if path.exists():
                with open(path, "r") as f:
                    evidence["reports"][key] = json.load(f)
                    
    # Reconstruction Audit
    audit_path = workspace_path / "reconstruction_audit.json"
    if audit_path.exists():
        with open(audit_path, "r") as f:
            evidence["reports"]["reconstruction_audit"] = json.load(f)
            
    # Cleanup stats
    # We need to find where cleaned mesh and its stats are.
    cleaned_dir = data_root / "cleaned" / session_id if session_id else None
    if cleaned_dir and cleaned_dir.exists():
        stats_path = cleaned_dir / "cleanup_stats.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                evidence["reports"]["cleanup_stats"] = json.load(f)
        
        meta_path = cleaned_dir / "normalized_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                evidence["reports"]["normalized_metadata"] = json.load(f)

    # Fusion Ablation Report (if exists in workspace or output)
    ablation_path = workspace_path / "ablation_report.json"
    if ablation_path.exists():
        with open(ablation_path, "r") as f:
            evidence["reports"]["fusion_ablation"] = json.load(f)

    # Collect Logs
    log_files = list(workspace_path.glob("*.log"))
    for attempt_dir in workspace_path.glob("attempt_*"):
        log_files.extend(list(attempt_dir.glob("*.log")))
        
    for log_path in log_files:
        rel_path = log_path.relative_to(workspace_path)
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
                    # Avoid infinite recursion or extremely large trees
                    if child.name == ".git" or child.name == "__pycache__":
                        continue
                    build_tree(child, indent + "  ")
            except PermissionError:
                tree_lines.append(f"{indent}  [Permission Denied]")
    
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

def generate_checklist(evidence: Dict[str, Any]) -> Dict[str, Any]:
    reports = evidence.get("reports", {})
    cap_report = reports.get("capture_report", {})
    ext_report = reports.get("extraction_report", {})
    val_report = reports.get("validation_report", {})
    audit = reports.get("reconstruction_audit", {})
    cleanup_stats = reports.get("cleanup_stats", {})
    export_metrics = reports.get("export_metrics", {})
    
    best_attempt = None
    if audit and audit.get("attempts") and audit.get("selected_best_index") is not None:
        best_attempt = audit["attempts"][audit["selected_best_index"]]
    
    metadata = best_attempt.get("metadata", {}) if best_attempt else {}
    
    # Checklist criteria
    checklist = {
        "capture_status": {
            "status": cap_report.get("overall_status") in ["sufficient", "warn"],
            "value": cap_report.get("overall_status"),
            "required": "PASS or WARN"
        },
        "accepted_frames": {
            "status": ext_report.get("saved_count", 0) >= 15,
            "value": ext_report.get("saved_count"),
            "required": ">= 15"
        },
        "dense_masks_match": {
            "status": bool(metadata.get("dense_mask_dimension_matches", False) and metadata.get("dense_mask_exact_filename_matches", False)),
            "value": f"dim_match={metadata.get('dense_mask_dimension_matches')}, file_match={metadata.get('dense_mask_exact_filename_matches')}",
            "required": "Exact/Dimension Match"
        },
        "fallback_white_ratio": {
            "status": metadata.get("dense_mask_fallback_white_ratio", 1.0) < 0.3,
            "value": metadata.get("dense_mask_fallback_white_ratio"),
            "required": "< 0.3"
        },
        "registered_image_count": {
            "status": best_attempt.get("registered_images", 0) >= 10 if best_attempt else False,
            "value": best_attempt.get("registered_images") if best_attempt else 0,
            "required": ">= 10"
        },
        "fused_point_count": {
            "status": best_attempt.get("dense_points_fused", 0) >= 5000 if best_attempt else False,
            "value": best_attempt.get("dense_points_fused") if best_attempt else 0,
            "required": ">= 5000"
        },
        "object_isolation_method": {
            "status": cleanup_stats.get("isolation", {}).get("object_isolation_method") in ["mask_guided", "hybrid_pc_mask"],
            "value": cleanup_stats.get("isolation", {}).get("object_isolation_method"),
            "required": "mask_guided or hybrid_pc_mask"
        },
        "isolation_confidence": {
            "status": cleanup_stats.get("isolation", {}).get("isolation_confidence", 0.0) >= 0.7,
            "value": cleanup_stats.get("isolation", {}).get("isolation_confidence"),
            "required": ">= 0.7"
        },
        "texture_count": {
            "status": export_metrics.get("texture_count", 0) > 0,
            "value": export_metrics.get("texture_count"),
            "required": "> 0"
        },
        "material_count": {
            "status": export_metrics.get("material_count", 0) > 0,
            "value": export_metrics.get("material_count"),
            "required": "> 0"
        },
        "texcoord_0_exists": {
            "status": export_metrics.get("all_textured_primitives_have_texcoord_0", False),
            "value": export_metrics.get("all_textured_primitives_have_texcoord_0"),
            "required": "True"
        },
        "texture_applied": {
            "status": export_metrics.get("texture_applied", False),
            "value": export_metrics.get("texture_applied"),
            "required": "True"
        },
        "glb_validation": {
            "status": val_report.get("final_decision") in ["pass", "review"],
            "value": val_report.get("final_decision"),
            "required": "PASS or REVIEW"
        }
    }
    
    return checklist

def generate_markdown_report(evidence: Dict[str, Any], output_path: Path):
    job_id = evidence["job_id"]
    checklist = evidence.get("delivery_checklist", {})
    
    lines = [
        f"# Reconstruction Evidence Report: {job_id}",
        f"**Exported At:** {evidence['exported_at']}",
        "",
        "## Delivery Checklist Summary",
        "| Criterion | Status | Value | Required |",
        "|-----------|--------|-------|----------|",
    ]
    
    all_pass = True
    for key, item in checklist.items():
        status_icon = "✅" if item["status"] else "❌"
        if not item["status"]:
            all_pass = False
        lines.append(f"| {key.replace('_', ' ').title()} | {status_icon} | {item['value']} | {item['required']} |")
    
    lines.append("")
    if all_pass:
        lines.append("> [!TIP]")
        lines.append("> Asset is DELIVERY READY.")
    else:
        lines.append("> [!CAUTION]")
        lines.append("> Asset FAILED one or more delivery criteria.")
        
    lines.append("")
    lines.append("## Configuration Snapshot")
    lines.append("```json")
    lines.append(json.dumps(evidence["config_snapshot"], indent=2))
    lines.append("```")
    
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
