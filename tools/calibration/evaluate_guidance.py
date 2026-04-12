#!/usr/bin/env python3
"""
tools/calibration/evaluate_guidance.py

CALIBRATION PHASE — Guidance Calibration Evaluator

WHAT THIS DOES
==============
Measures whether the GuidanceAggregator's decisions match labeled expectations:
  - Does next_action match expected_action?
  - Is severity too harsh or too weak?
  - Are message codes noisy (too many per fixture)?

This is NOT about rewriting guidance copy.
It is about measuring guidance BEHAVIOR against labeled ground truth.

USAGE
=====
    python tools/calibration/evaluate_guidance.py

    python tools/calibration/evaluate_guidance.py \\
        --fixtures tools/calibration/fixtures/real_captures/real_capture_fixtures.json

    python tools/calibration/evaluate_guidance.py --output guidance_report.md

EVALUATION DIMENSIONS
=====================
1. Action Match:
   Does guidance.should_recapture / is_ready_for_review match expected_action?
   - expected_action=accept + should_recapture=False + is_ready_for_review=True  -> CORRECT
   - expected_action=accept + should_recapture=True                              -> OVERCAUTIOUS (FP)
   - expected_action=recapture + should_recapture=False                          -> MISSED (FN)
   - expected_action=review + is_ready_for_review=True                           -> CORRECT

2. Severity Balance:
   For fixtures where expected_action=accept:
     Any CRITICAL severity message -> OVERCAUTIOUS
   For fixtures where expected_action=recapture:
     No CRITICAL severity message  -> WEAK

3. Message Density:
   More than N messages for a single fixture -> NOISY
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_FIXTURES = [
    Path(__file__).parent / "fixtures" / "coverage_fixtures.json",
    Path(__file__).parent / "fixtures" / "real_captures" / "real_capture_fixtures.json",
]

_MAX_MESSAGES_BEFORE_NOISY = 5


# ─────────────────────────────────────────────────────────────────────────────
# Guidance generation
# ─────────────────────────────────────────────────────────────────────────────

def _run_guidance(fixture: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Runs GuidanceAggregator.generate_guidance() on fixture data.
    Returns the guidance dict or None on import failure.
    """
    try:
        from modules.operations.guidance import GuidanceAggregator
        from modules.shared_contracts.lifecycle import AssetStatus

        cov = fixture.get("coverage_report") or {}
        val = fixture.get("validation_report")
        session_meta = fixture.get("_pipeline_status", "validated")

        # Map session status
        status_map = {
            "published":          AssetStatus.PUBLISHED,
            "validated":          AssetStatus.VALIDATED,
            "failed":             AssetStatus.FAILED,
            "recapture_required": AssetStatus.RECAPTURE_REQUIRED,
            "captured":           AssetStatus.CAPTURED,
            "reconstructed":      AssetStatus.RECONSTRUCTED,
            "cleaned":            AssetStatus.CLEANED,
            "exported":           AssetStatus.EXPORTED,
        }
        status_str = str(session_meta or "validated").lower()
        status = status_map.get(status_str, AssetStatus.VALIDATED)

        agg = GuidanceAggregator()
        guidance = agg.generate_guidance(
            session_id=fixture.get("session_id", "fixture"),
            status=status,
            coverage_report=cov if cov else None,
            validation_report=val,
        )

        messages = guidance.messages

        return {
            "should_recapture":    guidance.should_recapture,
            "is_ready_for_review": guidance.is_ready_for_review,
            "next_action":         guidance.next_action,
            "message_count":       len(messages),
            "message_codes":       [m["code"] for m in messages],
            "has_critical":        any(
                "critical" in str(m.get("severity", "")).lower()
                for m in messages
            ),
            "has_warning":         any(
                "warning" in str(m.get("severity", "")).lower()
                for m in messages
            ),
        }
    except ImportError as e:
        print(f"[WARN] Cannot run live guidance (import error: {e}). "
              "Using stored guidance_report from fixture.", file=sys.stderr)
        return None


def _use_stored_guidance(fixture: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Fall back to stored guidance_report field in fixture if live evaluation fails.
    """
    g = fixture.get("guidance_report")
    if g is None:
        return None
    return {
        "should_recapture":    g.get("should_recapture", False),
        "is_ready_for_review": g.get("is_ready_for_review", False),
        "next_action":         g.get("next_action", ""),
        "message_count":       len(g.get("message_codes", [])),
        "message_codes":       g.get("message_codes", []),
        "has_critical":        False,  # Not known from stored
        "has_warning":         False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-fixture evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_fixture_guidance(
    fixture: Dict[str, Any],
    live: bool = True,
) -> Dict[str, Any]:
    fid = fixture.get("fixture_id", "?")
    expected_action = fixture.get("expected_action", "")
    label = fixture.get("label", "")
    source = fixture.get("source", "synthetic")

    # Get guidance
    guidance = _run_guidance(fixture) if live else None
    if guidance is None:
        guidance = _use_stored_guidance(fixture)
    if guidance is None:
        return {
            "fixture_id": fid,
            "expected_action": expected_action,
            "label": label,
            "source": source,
            "status": "no_guidance_data",
            "action_match": None,
            "severity_ok": None,
            "is_noisy": None,
            "issues": ["No guidance data available (run through pipeline first)"],
        }

    # ── 1. Action Match ───────────────────────────────────────────────────────
    should_recapture = guidance["should_recapture"]
    is_ready = guidance["is_ready_for_review"]

    if expected_action in {"accept", "review"}:
        action_match = not should_recapture
    elif expected_action == "recapture":
        action_match = should_recapture
    elif expected_action == "fail":
        action_match = should_recapture  # fail also triggers recapture flag
    else:
        action_match = None

    # ── 2. Severity Balance ───────────────────────────────────────────────────
    has_critical = guidance["has_critical"]
    if expected_action in {"accept"}:
        # CRITICAL message on a good capture -> overcautious
        severity_ok = not has_critical
    elif expected_action in {"recapture", "fail"}:
        # Should have at least a CRITICAL message
        severity_ok = has_critical
    else:
        severity_ok = True  # review -> neutral

    # ── 3. Message Density ────────────────────────────────────────────────────
    msg_count = guidance["message_count"]
    is_noisy = msg_count > _MAX_MESSAGES_BEFORE_NOISY

    # ── Issues list ───────────────────────────────────────────────────────────
    issues = []
    if action_match is False:
        if expected_action in {"accept", "review"} and should_recapture:
            issues.append(f"OVERCAUTIOUS: guidance says recapture for a '{expected_action}' case")
        elif expected_action in {"recapture", "fail"} and not should_recapture:
            issues.append(f"MISSED: guidance did not flag recapture for '{expected_action}' case")
    if severity_ok is False:
        if expected_action in {"accept"} and has_critical:
            issues.append("OVERCAUTIOUS_SEVERITY: CRITICAL message for an acceptable capture")
        elif expected_action in {"recapture", "fail"} and not has_critical:
            issues.append("WEAK_SEVERITY: no CRITICAL message for a recapture/fail case")
    if is_noisy:
        issues.append(f"NOISY: {msg_count} messages (limit: {_MAX_MESSAGES_BEFORE_NOISY})")

    overall_ok = (
        action_match is not False
        and severity_ok is not False
        and not is_noisy
    )

    return {
        "fixture_id":         fid,
        "expected_action":    expected_action,
        "label":              label,
        "source":             source,
        "should_recapture":   should_recapture,
        "is_ready_for_review": is_ready,
        "next_action":        guidance["next_action"][:80],
        "message_count":      msg_count,
        "message_codes":      guidance["message_codes"],
        "has_critical":       has_critical,
        "action_match":       action_match,
        "severity_ok":        severity_ok,
        "is_noisy":           is_noisy,
        "overall_ok":         overall_ok,
        "issues":             issues,
        "status":             "evaluated",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_guidance_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [r for r in results if r["status"] == "evaluated"]
    if not evaluated:
        return {"total": 0, "evaluated": 0}

    total = len(evaluated)
    action_correct = sum(1 for r in evaluated if r["action_match"] is True)
    severity_ok = sum(1 for r in evaluated if r["severity_ok"] is True)
    noisy = sum(1 for r in evaluated if r["is_noisy"])
    overcautious = sum(
        1 for r in evaluated
        if r["action_match"] is False
        and r["expected_action"] in {"accept", "review"}
    )
    missed = sum(
        1 for r in evaluated
        if r["action_match"] is False
        and r["expected_action"] in {"recapture", "fail"}
    )
    avg_messages = sum(r["message_count"] for r in evaluated) / total

    return {
        "total":          total,
        "action_correct": action_correct,
        "action_accuracy": round(action_correct / total, 4),
        "severity_ok":    severity_ok,
        "severity_accuracy": round(severity_ok / total, 4),
        "noisy_count":    noisy,
        "overcautious":   overcautious,
        "missed":         missed,
        "avg_messages":   round(avg_messages, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────────────────────────────────────

def _render_markdown(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> str:
    lines = [
        "# Guidance Calibration Report",
        "",
        "## Summary",
        "| Metric | Value |",
        "|---|---|",
        f"| Total fixtures evaluated | {metrics.get('total', 0)} |",
        f"| Action match accuracy | {metrics.get('action_accuracy', 0):.2%} |",
        f"| Severity accuracy | {metrics.get('severity_accuracy', 0):.2%} |",
        f"| Overcautious (FP) | {metrics.get('overcautious', 0)} |",
        f"| Missed (FN) | {metrics.get('missed', 0)} |",
        f"| Noisy (>5 messages) | {metrics.get('noisy_count', 0)} |",
        f"| Avg messages per fixture | {metrics.get('avg_messages', 0):.1f} |",
        "",
        "## Per-Fixture Results",
        "| ID | Expected | Recapture? | Action? | Severity? | Noisy? | OK? |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in results:
        if r["status"] != "evaluated":
            lines.append(f"| {r['fixture_id']} | {r['expected_action']} "
                         "| - | - | - | - | [NO DATA] |")
            continue
        ok = "[OK]" if r["overall_ok"] else "[FAIL]"
        lines.append(
            f"| {r['fixture_id']} | {r['expected_action']} "
            f"| {r['should_recapture']} "
            f"| {'[OK]' if r['action_match'] else '[FAIL]'} "
            f"| {'[OK]' if r['severity_ok'] else '[FAIL]'} "
            f"| {'yes' if r['is_noisy'] else 'no'} "
            f"| {ok} |"
        )

    lines += ["", "### Issues"]
    problem_results = [r for r in results if r.get("issues")]
    if not problem_results:
        lines.append("_No guidance issues detected._")
    else:
        for r in problem_results:
            lines.append(f"\n**{r['fixture_id']}** (expected: {r['expected_action']})")
            for iss in r["issues"]:
                lines.append(f"  - {iss}")
            lines.append(f"  - Codes: {r.get('message_codes', [])}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_guidance_evaluation(
    fixture_paths: List[Path],
    output_path: Optional[Path] = None,
    fmt: str = "both",
) -> Dict[str, Any]:
    all_fixtures = []
    for p in fixture_paths:
        if not p.exists():
            print(f"[WARN] Skipping missing fixture file: {p}", file=sys.stderr)
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        all_fixtures.extend(data)
        print(f"[INFO] Loaded {len(data)} fixtures from {p.name}", file=sys.stderr)

    results = [_evaluate_fixture_guidance(f) for f in all_fixtures]
    metrics = _compute_guidance_metrics(results)

    report = {
        "metrics": metrics,
        "results": results,
    }

    md = _render_markdown(results, metrics)

    if fmt in {"both", "markdown"}:
        print(md)
        print()

    if fmt in {"both", "json"}:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".md":
            output_path.write_text(md, encoding="utf-8")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate guidance messaging behavior against labeled fixtures."
    )
    parser.add_argument("--fixtures", nargs="*", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", dest="fmt",
                        choices=["json", "markdown", "both"], default="both")
    args = parser.parse_args()

    fixture_paths = args.fixtures or _DEFAULT_FIXTURES
    run_guidance_evaluation(fixture_paths, output_path=args.output, fmt=args.fmt)


if __name__ == "__main__":
    main()
