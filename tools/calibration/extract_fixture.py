#!/usr/bin/env python3
"""
tools/calibration/extract_fixture.py

CALIBRATION PHASE — Real-World Fixture Extraction Utility

WHAT THIS DOES
==============
Extracts a calibration fixture from an existing pipeline session directory.
Reads coverage_report.json, validation_report.json, guidance_report.json,
and session JSON metadata, then produces a single fixture entry that can be
added to the calibration dataset.

This eliminates the need to hand-author every JSON fixture.
Operators extract, then add a label and observed_issue_classes.

USAGE
=====
    # Extract a fixture from a session by session_id:
    python tools/calibration/extract_fixture.py --session-id cap_c075957c --data-root data

    # Extract and immediately add a label:
    python tools/calibration/extract_fixture.py \\
        --session-id cap_c075957c \\
        --data-root data \\
        --label good \\
        --expected-action accept \\
        --product-type sneaker \\
        --capture-environment studio \\
        --notes "First production scan, clean result"

    # Save directly to the real_captures fixture file:
    python tools/calibration/extract_fixture.py \\
        --session-id cap_c075957c \\
        --data-root data \\
        --label good \\
        --expected-action accept \\
        --append tools/calibration/fixtures/real_captures/real_capture_fixtures.json

    # Extract all sessions from a data directory:
    python tools/calibration/extract_fixture.py --extract-all --data-root data --output-dir tools/calibration/fixtures/real_captures/batch/

WORKFLOW
========
1. Operator runs a capture session through the pipeline.
2. After the session reaches VALIDATED or FAILED:
      python tools/calibration/extract_fixture.py \\
          --session-id <id> --data-root data \\
          --label <good|recapture|bad|noisy|insufficient> \\
          --expected-action <accept|recapture|review|fail>
3. Output fixture is saved and can be reviewed/edited.
4. Add to the fixture dataset file for evaluation.

OUTPUT FORMAT
=============
A single JSON object conforming to dataset_schema.json.
Missing fields (coverage_report, validation_report) are noted as null with warnings.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root on path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_VALID_LABELS = {"good", "recapture", "bad", "noisy", "insufficient"}
_VALID_ACTIONS = {"accept", "recapture", "review", "fail"}
_VALID_ENVS = {"studio", "office", "outdoor", "handheld", "turntable", "unknown"}
_VALID_ISSUE_CLASSES = {
    "low_diversity", "narrow_motion", "scale_flat", "too_few_frames",
    "masking_fallback", "low_confidence", "ml_unavailable",
    "contamination_high", "contamination_fragmented", "texture_failure",
    "blur", "overexposure", "underexposure", "object_too_small",
    "object_too_large", "object_clipped", "background_cluttered", "none",
}


# ─────────────────────────────────────────────────────────────────────────────
# Session path resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_paths(data_root: Path, session_id: str) -> Dict[str, Optional[Path]]:
    """
    Resolves expected artifact paths for a session.
    Returns dict of path_name -> Path (or None if not found).
    """
    sessions_dir = data_root / "sessions"
    captures_dir = data_root / "captures"

    session_file = sessions_dir / f"{session_id}.json"
    reports_dir = captures_dir / session_id / "reports"

    return {
        "session_file":        session_file if session_file.exists() else None,
        "coverage_report":     _try(reports_dir / "coverage_report.json"),
        "validation_report":   _try(reports_dir / "validation_report.json"),
        "guidance_report":     _try(reports_dir / "guidance_report.json"),
        "export_metrics":      _try(reports_dir / "export_metrics.json"),
        "cleanup_stats":       _find_cleanup_stats(data_root, session_id),
    }


def _try(p: Path) -> Optional[Path]:
    return p if p.exists() else None


def _find_cleanup_stats(data_root: Path, session_id: str) -> Optional[Path]:
    """
    Cleanup stats may be in captures/<id>/cleaned/ or similar.
    Try a few likely locations.
    """
    candidates = [
        data_root / "captures" / session_id / "cleaned" / "cleanup_stats.json",
        data_root / "captures" / session_id / "reports" / "cleanup_stats.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Report loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Failed to load {path}: {e}", file=sys.stderr)
        return None


def _load_session_meta(path: Optional[Path]) -> Dict[str, Any]:
    data = _load_json(path) or {}
    return {
        "session_id":    data.get("session_id", "unknown"),
        "product_id":    data.get("product_id", "unknown"),
        "operator_id":   data.get("operator_id", "unknown"),
        "status":        data.get("status", "unknown"),
        "publish_state": data.get("publish_state"),
        "failure_reason": data.get("failure_reason"),
        "created_at":    data.get("created_at"),
        "coverage_score": data.get("coverage_score", 0.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fixture ID generation
# ─────────────────────────────────────────────────────────────────────────────

def _make_fixture_id(session_id: str) -> str:
    slug = session_id[:12].replace(" ", "_").replace("-", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"real_{slug}_{ts}"


# ─────────────────────────────────────────────────────────────────────────────
# Auto-infer issue classes from reports
# ─────────────────────────────────────────────────────────────────────────────

def _infer_issue_classes(
    coverage_report: Optional[Dict[str, Any]],
    validation_report: Optional[Dict[str, Any]],
) -> List[str]:
    """
    Automatically infers observed_issue_classes from report content.
    Operator should review and correct these suggestions.
    """
    issues = []

    if coverage_report:
        reasons = " ".join(coverage_report.get("reasons", [])).lower()
        fallback_frames = int(coverage_report.get("fallback_frames", 0))
        readable = int(coverage_report.get("readable_frames", 1))

        if "insufficient viewpoint diversity" in reasons or "unique view" in reasons:
            issues.append("low_diversity")
        if "narrow" in reasons or "object motion" in reasons:
            issues.append("narrow_motion")
        if "scale" in reasons or "shape variation" in reasons:
            issues.append("scale_flat")
        if "few readable" in reasons or "too few" in reasons:
            issues.append("too_few_frames")
        if fallback_frames > max(1, readable * 0.4):
            issues.append("masking_fallback")
        if "low semantic confidence" in reasons or "low confidence" in reasons:
            issues.append("low_confidence")
        if coverage_report.get("ml_segmentation_unavailable"):
            issues.append("ml_unavailable")

    if validation_report:
        contamination = float(validation_report.get("contamination_score", 0.0))
        component_share = float(validation_report.get("largest_component_share", 1.0))
        component_count = int(validation_report.get("component_count", 1))
        integrity = validation_report.get("contamination_report", {})

        if contamination > 0.4:
            issues.append("contamination_high")
        if component_share < 0.80 and component_count > 2:
            issues.append("contamination_fragmented")
        if isinstance(integrity, dict) and integrity.get("texture_uv_integrity") == "fail":
            issues.append("texture_failure")

    return issues if issues else ["none"]


# ─────────────────────────────────────────────────────────────────────────────
# Auto-infer expected_action from session/report state
# ─────────────────────────────────────────────────────────────────────────────

def _infer_expected_action(
    session_meta: Dict[str, Any],
    coverage_report: Optional[Dict[str, Any]],
    validation_report: Optional[Dict[str, Any]],
) -> str:
    """
    Suggests expected_action from the pipeline's actual decision.
    Operator should confirm/override this.
    """
    status = session_meta.get("status", "unknown")
    publish_state = session_meta.get("publish_state")

    if status == "recapture_required":
        return "recapture"
    if status == "failed":
        return "fail"
    if status == "published":
        return "accept"
    if validation_report:
        dec = validation_report.get("final_decision", "")
        if dec == "pass":
            return "accept"
        if dec == "review":
            return "review"
        if dec == "fail":
            return "fail"
    if coverage_report:
        if coverage_report.get("overall_status") == "sufficient":
            return "accept"
        return "recapture"
    return "review"


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────────────────────

def extract_fixture(
    session_id: str,
    data_root: Path,
    label: Optional[str] = None,
    expected_action: Optional[str] = None,
    product_type: str = "unknown",
    capture_environment: str = "unknown",
    notes: str = "",
    issue_classes: Optional[List[str]] = None,
    operator_id: str = "unknown",
) -> Dict[str, Any]:
    """
    Main extraction function. Returns a fixture dict.
    """
    print(f"[INFO] Extracting fixture for session: {session_id}", file=sys.stderr)

    paths = _resolve_paths(data_root, session_id)
    session_meta = _load_session_meta(paths["session_file"])
    coverage_report = _load_json(paths["coverage_report"])
    validation_report = _load_json(paths["validation_report"])
    guidance_report = _load_json(paths["guidance_report"])

    # Diagnostics
    if coverage_report is None:
        print(
            f"  [WARN] coverage_report.json not found at "
            f"{data_root / 'captures' / session_id / 'reports'}",
            file=sys.stderr,
        )
    if validation_report is None:
        print(
            f"  [WARN] validation_report.json not found — "
            f"session may not have reached VALIDATED status",
            file=sys.stderr,
        )

    # Auto-infer where not provided
    inferred_issues = _infer_issue_classes(coverage_report, validation_report)
    inferred_action = _infer_expected_action(session_meta, coverage_report, validation_report)

    final_label = label or _suggest_label(session_meta, coverage_report)
    final_action = expected_action or inferred_action
    final_issues = issue_classes or inferred_issues

    fixture = {
        "fixture_id":          _make_fixture_id(session_id),
        "description":         notes or f"Extracted from session {session_id}",
        "session_id":          session_meta["session_id"],
        "product_type":        product_type,
        "capture_environment": capture_environment,
        "label":               final_label,
        "expected_action":     final_action,
        "observed_issue_classes": final_issues,
        "source":              "extracted",
        "extracted_at":        datetime.now(timezone.utc).isoformat(),
        "coverage_report":     coverage_report or {
            "overall_status": "unknown",
            "coverage_score": 0.0,
            "reasons": ["[MISSING: coverage_report.json not found]"],
        },
        "validation_report":   validation_report,
        "guidance_report":     (
            {
                "next_action":      guidance_report.get("next_action"),
                "should_recapture": guidance_report.get("should_recapture"),
                "is_ready_for_review": guidance_report.get("is_ready_for_review"),
                "message_codes":    [
                    m.get("code") for m in guidance_report.get("messages", [])
                ],
            }
            if guidance_report else None
        ),
        "notes":               notes,
        "operator_id":         operator_id,
        "labeled_at":          datetime.now(timezone.utc).isoformat(),
        "_pipeline_status":    session_meta.get("status"),
        "_inferred_action":    inferred_action,
        "_inferred_issues":    inferred_issues,
    }

    print(f"  [INFO] Inferred label:    {final_label}", file=sys.stderr)
    print(f"  [INFO] Inferred action:   {final_action}", file=sys.stderr)
    print(f"  [INFO] Inferred issues:   {final_issues}", file=sys.stderr)
    if label is None:
        print(
            "  [WARN] No --label provided. Inferred label used — please REVIEW and set manually.",
            file=sys.stderr,
        )

    return fixture


def _suggest_label(
    session_meta: Dict[str, Any],
    coverage_report: Optional[Dict[str, Any]],
) -> str:
    status = session_meta.get("status", "unknown")
    if status == "published":
        return "good"
    if status in {"failed", "recapture_required"}:
        return "recapture"
    if coverage_report and coverage_report.get("overall_status") == "sufficient":
        return "good"
    return "recapture"


# ─────────────────────────────────────────────────────────────────────────────
# Batch extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_all(data_root: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract fixtures for all sessions found in data_root/sessions/.
    Saves one fixture file per session.
    Returns list of all extracted fixtures.
    """
    sessions_dir = data_root / "sessions"
    if not sessions_dir.exists():
        print(f"[ERROR] Sessions directory not found: {sessions_dir}", file=sys.stderr)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    all_fixtures = []

    for f in sorted(sessions_dir.glob("*.json")):
        session_id = f.stem
        try:
            fixture = extract_fixture(
                session_id=session_id,
                data_root=data_root,
                operator_id="batch_extract",
            )
            out_file = output_dir / f"{fixture['fixture_id']}.json"
            with open(out_file, "w", encoding="utf-8") as fp:
                json.dump(fixture, fp, indent=2, ensure_ascii=False)
            all_fixtures.append(fixture)
            print(f"  [OK] {session_id} -> {out_file.name}", file=sys.stderr)
        except Exception as e:
            print(f"  [ERROR] {session_id}: {e}", file=sys.stderr)

    print(f"\n[INFO] Extracted {len(all_fixtures)} fixture(s) to {output_dir}", file=sys.stderr)
    return all_fixtures


# ─────────────────────────────────────────────────────────────────────────────
# Append to existing fixture file
# ─────────────────────────────────────────────────────────────────────────────

def append_to_fixture_file(fixture: Dict[str, Any], fixture_file: Path) -> None:
    """
    Appends a fixture to an existing fixture JSON array file.
    Creates the file if it doesn't exist.
    """
    existing = []
    if fixture_file.exists():
        with open(fixture_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        # Check for duplicate fixture_id
        existing_ids = {e.get("fixture_id") for e in existing}
        if fixture["fixture_id"] in existing_ids:
            print(
                f"[WARN] fixture_id '{fixture['fixture_id']}' already exists in {fixture_file}. "
                "Not appending to avoid duplicates.",
                file=sys.stderr,
            )
            return

    existing.append(fixture)
    with open(fixture_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(
        f"[INFO] Appended fixture '{fixture['fixture_id']}' to {fixture_file} "
        f"(total: {len(existing)})",
        file=sys.stderr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract calibration fixtures from existing pipeline sessions."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Single session extraction
    single = subparsers.add_parser("extract", help="Extract fixture from a single session")
    single.add_argument("--session-id", required=True, help="Session ID to extract")
    single.add_argument("--data-root", type=Path, default=Path("data"),
                        help="Data root directory (default: data)")
    single.add_argument("--label", choices=sorted(_VALID_LABELS),
                        help="Ground truth label for this fixture")
    single.add_argument("--expected-action", choices=sorted(_VALID_ACTIONS),
                        help="Expected pipeline action")
    single.add_argument("--product-type", default="unknown",
                        help="Type of product being scanned")
    single.add_argument("--capture-environment", choices=sorted(_VALID_ENVS),
                        default="unknown", help="Capture environment")
    single.add_argument("--issues", nargs="*", choices=sorted(_VALID_ISSUE_CLASSES),
                        help="Observed issue classes (space-separated)")
    single.add_argument("--notes", default="", help="Operator notes")
    single.add_argument("--operator-id", default="unknown", help="Operator ID")
    single.add_argument("--output", type=Path, default=None,
                        help="Save fixture to this JSON file")
    single.add_argument("--append", type=Path, default=None,
                        help="Append fixture to this existing fixture JSON array file")

    # Batch extraction
    batch = subparsers.add_parser("extract-all", help="Extract fixtures from all sessions")
    batch.add_argument("--data-root", type=Path, default=Path("data"))
    batch.add_argument("--output-dir", type=Path,
                       default=Path("tools/calibration/fixtures/real_captures/batch"))

    args = parser.parse_args()

    # Default to 'extract' if run directly without subcommand (for simple usage)
    if args.command is None:
        # Allow old-style --session-id without subcommand for convenience
        parser_compat = argparse.ArgumentParser()
        parser_compat.add_argument("--session-id")
        parser_compat.add_argument("--data-root", type=Path, default=Path("data"))
        parser_compat.add_argument("--label", default=None)
        parser_compat.add_argument("--expected-action", default=None)
        parser_compat.add_argument("--product-type", default="unknown")
        parser_compat.add_argument("--capture-environment", default="unknown")
        parser_compat.add_argument("--issues", nargs="*")
        parser_compat.add_argument("--notes", default="")
        parser_compat.add_argument("--operator-id", default="unknown")
        parser_compat.add_argument("--output", type=Path, default=None)
        parser_compat.add_argument("--append", type=Path, default=None)
        parser_compat.add_argument("--extract-all", action="store_true")
        parser_compat.add_argument("--output-dir", type=Path,
                                   default=Path("tools/calibration/fixtures/real_captures/batch"))
        args = parser_compat.parse_args()

        if getattr(args, "extract_all", False):
            extract_all(args.data_root, args.output_dir)
            return
        if not args.session_id:
            parser_compat.print_help()
            sys.exit(1)

    if getattr(args, "command", None) == "extract-all":
        extract_all(args.data_root, args.output_dir)
        return

    session_id = args.session_id
    fixture = extract_fixture(
        session_id=session_id,
        data_root=args.data_root,
        label=args.label,
        expected_action=getattr(args, "expected_action", None),
        product_type=getattr(args, "product_type", "unknown"),
        capture_environment=getattr(args, "capture_environment", "unknown"),
        notes=getattr(args, "notes", ""),
        issue_classes=getattr(args, "issues", None),
        operator_id=getattr(args, "operator_id", "unknown"),
    )

    # Output the fixture JSON
    output = getattr(args, "output", None)
    append = getattr(args, "append", None)

    if append:
        append_to_fixture_file(fixture, append)
    elif output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(fixture, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Fixture saved to {output}", file=sys.stderr)
    else:
        # Print to stdout
        print(json.dumps(fixture, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
