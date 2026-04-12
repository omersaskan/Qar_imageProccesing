#!/usr/bin/env python3
"""
tools/calibration/evaluate_coverage_heuristics.py

SPRINT 3 — TICKET-009: Ground-truth-oriented calibration support

WHAT THIS DOES
==============
Runs the CoverageAnalyzer decision logic against a set of labeled fixtures and
produces a confusion-style evaluation report.

This is NOT a training script. It is a calibration evaluation harness that lets
the team see how the current heuristic thresholds perform on labeled examples.

USAGE
=====
    # Run on the bundled fixtures (default):
    python tools/calibration/evaluate_coverage_heuristics.py

    # Run on a custom fixture file:
    python tools/calibration/evaluate_coverage_heuristics.py --fixtures path/to/your_fixtures.json

    # Save the report:
    python tools/calibration/evaluate_coverage_heuristics.py --output report.json

    # Show markdown summary only:
    python tools/calibration/evaluate_coverage_heuristics.py --format markdown

FIXTURE FORMAT
==============
Each fixture is a JSON object with at minimum:
  - fixture_id    (str)
  - label         (str): "good" | "recapture" | "bad" | "noisy" | "insufficient"
  - coverage_report (dict): the exact dict that CoverageAnalyzer.analyze_coverage() returns

See tools/calibration/fixtures/coverage_fixtures.json for examples.
See tools/calibration/fixture_schema.json for the full schema.

DECISION MAPPING
================
The heuristic produces "sufficient" or "insufficient" in overall_status.
This is mapped to a binary decision for evaluation:
  "sufficient"   → predicted "good"
  "insufficient" → predicted "recapture"

Ground truth labels are mapped as:
  "good"                     → expected "good"
  "recapture" | "bad" | "noisy" | "insufficient" → expected "recapture"

This gives us a 2x2 confusion matrix:
  TP: correctly predicted recapture
  TN: correctly predicted good
  FP: predicted recapture but was actually good
  FN: predicted good but was actually recapture/bad

CALIBRATION GUIDANCE
====================
If precision is low  → Too many false positives (heuristic is too strict).
                        Consider relaxing min_unique_views or min_readable_frames.
If recall is low     → Too many false negatives (heuristic misses bad captures).
                        Consider tightening fallback_frame ratio or confidence thresholds.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running from repo root without installing
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_FIXTURES = Path(__file__).parent / "fixtures" / "coverage_fixtures.json"


# ─────────────────────────────────────────────────────────────────────────────
# Label normalization
# ─────────────────────────────────────────────────────────────────────────────

_RECAPTURE_LABELS = {"recapture", "bad", "noisy", "insufficient"}
_GOOD_LABELS = {"good"}


def _normalize_expected(label: str) -> str:
    """Map fixture label to binary expected class."""
    if label.lower() in _GOOD_LABELS:
        return "good"
    elif label.lower() in _RECAPTURE_LABELS:
        return "recapture"
    raise ValueError(f"Unknown label: {label!r}")


def _normalize_predicted(coverage_report: Dict[str, Any]) -> str:
    """Map CoverageAnalyzer output to binary predicted class."""
    status = coverage_report.get("overall_status", "insufficient")
    return "good" if status == "sufficient" else "recapture"


# ─────────────────────────────────────────────────────────────────────────────
# Per-fixture evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_fixture(fixture: Dict[str, Any]) -> Dict[str, Any]:
    fid = fixture["fixture_id"]
    label = fixture["label"]
    report = fixture["coverage_report"]
    desc = fixture.get("description", "")

    expected = _normalize_expected(label)
    predicted = _normalize_predicted(report)
    correct = expected == predicted

    reasons = report.get("reasons", [])
    coverage_score = float(report.get("coverage_score", 0.0))
    unique_views = int(report.get("unique_views", 0))

    return {
        "fixture_id": fid,
        "description": desc,
        "raw_label": label,
        "expected": expected,
        "predicted": predicted,
        "correct": correct,
        "coverage_score": round(coverage_score, 3),
        "unique_views": unique_views,
        "reasons": reasons,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix and metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = tn = fp = fn = 0
    for r in results:
        exp = r["expected"]
        pred = r["predicted"]
        if exp == "recapture" and pred == "recapture":
            tp += 1
        elif exp == "good" and pred == "good":
            tn += 1
        elif exp == "good" and pred == "recapture":
            fp += 1
        elif exp == "recapture" and pred == "good":
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "total": total,
        "correct": tp + tn,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Calibration advice
# ─────────────────────────────────────────────────────────────────────────────

def _calibration_advice(metrics: Dict[str, Any]) -> List[str]:
    advice = []
    precision = metrics["precision"]
    recall = metrics["recall"]
    fp = metrics["fp"]
    fn = metrics["fn"]

    if precision < 0.70 and fp > 0:
        advice.append(
            f"PRECISION LOW ({precision:.2f}): {fp} good capture(s) incorrectly flagged for recapture. "
            "Consider relaxing min_unique_views or min_readable_frames in CoverageConfig."
        )
    if recall < 0.80 and fn > 0:
        advice.append(
            f"RECALL LOW ({recall:.2f}): {fn} bad/noisy capture(s) passed as 'sufficient'. "
            "Consider tightening fallback_frame ratio thresholds or min_unique_views."
        )
    if not advice:
        advice.append(
            f"Heuristics performing well (precision={precision:.2f}, recall={recall:.2f}). "
            "No immediate threshold adjustments recommended."
        )
    return advice


# ─────────────────────────────────────────────────────────────────────────────
# Report rendering
# ─────────────────────────────────────────────────────────────────────────────

def _render_markdown(results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> str:
    advice = _calibration_advice(metrics)
    lines = [
        "# Coverage Heuristic Calibration Report",
        "",
        "## Summary Metrics",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total fixtures | {metrics['total']} |",
        f"| Correct | {metrics['correct']} |",
        f"| Accuracy | {metrics['accuracy']:.2%} |",
        f"| Precision | {metrics['precision']:.2%} |",
        f"| Recall | {metrics['recall']:.2%} |",
        f"| F1 | {metrics['f1']:.2%} |",
        "",
        "## Confusion Matrix",
        f"| | Predicted: recapture | Predicted: good |",
        f"|---|---|---|",
        f"| **Expected: recapture** | TP = {metrics['tp']} | FN = {metrics['fn']} |",
        f"| **Expected: good**      | FP = {metrics['fp']} | TN = {metrics['tn']} |",
        "",
        "## Calibration Advice",
    ]
    for a in advice:
        lines.append(f"- {a}")
    lines += [
        "",
        "## Per-Fixture Results",
        "| ID | Label | Expected | Predicted | Score | Unique Views | OK? |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ok = "[OK]" if r["correct"] else "[FAIL]"
        lines.append(
            f"| {r['fixture_id']} | {r['raw_label']} | {r['expected']} "
            f"| {r['predicted']} | {r['coverage_score']:.2f} "
            f"| {r['unique_views']} | {ok} |"
        )
    lines += ["", "### Failure Details"]
    failed = [r for r in results if not r["correct"]]
    if not failed:
        lines.append("_No misclassifications — all fixtures predicted correctly._")
    else:
        for r in failed:
            lines.append(
                f"\n**{r['fixture_id']}** ({r['raw_label']}) → "
                f"predicted `{r['predicted']}`, expected `{r['expected']}`"
            )
            for reason in r["reasons"]:
                lines.append(f"  - {reason}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    fixtures_path: Path,
    output_path: Optional[Path] = None,
    fmt: str = "both",
) -> Dict[str, Any]:
    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    results = [_evaluate_fixture(fix) for fix in fixtures]
    metrics = _compute_metrics(results)
    advice = _calibration_advice(metrics)

    report = {
        "fixtures_file": str(fixtures_path),
        "metrics": metrics,
        "advice": advice,
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
        print(f"\nReport saved to: {output_path}", file=sys.stderr)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate coverage heuristics against labeled fixtures."
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=_DEFAULT_FIXTURES,
        help=f"Path to fixture JSON file (default: {_DEFAULT_FIXTURES})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save the report to this path (.json or .md)",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format (default: both)",
    )
    args = parser.parse_args()

    if not args.fixtures.exists():
        print(f"ERROR: Fixtures file not found: {args.fixtures}", file=sys.stderr)
        sys.exit(1)

    run_evaluation(args.fixtures, output_path=args.output, fmt=args.fmt)


if __name__ == "__main__":
    main()
