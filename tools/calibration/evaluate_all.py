#!/usr/bin/env python3
"""
tools/calibration/evaluate_all.py

CALIBRATION PHASE — Unified Evaluation Harness

WHAT THIS DOES
==============
Evaluates heuristic decisions against ALL labeled fixtures:
  - Synthetic fixtures (Sprint 3)
  - Real-capture fixtures (this phase)
  - Extracted fixtures

Provides:
  1. Confusion matrix (TP/TN/FP/FN)
  2. Per-issue-class failure breakdown
  3. Per-source breakdown (synthetic vs real vs extracted)
  4. Per-product_type breakdown
  5. False positive / false negative pattern analysis
  6. Threshold tuning recommendations (surfaced as suggestions only)

USAGE
=====
    # Evaluate all fixture files in default directories:
    python tools/calibration/evaluate_all.py

    # Specific fixture files:
    python tools/calibration/evaluate_all.py \\
        --fixtures tools/calibration/fixtures/coverage_fixtures.json \\
                   tools/calibration/fixtures/real_captures/real_capture_fixtures.json

    # Save markdown report:
    python tools/calibration/evaluate_all.py --output report.md

    # JSON output only:
    python tools/calibration/evaluate_all.py --format json

DECISION MAPPING
================
Coverage heuristic:
  overall_status == "sufficient"   -> predicted "accept"
  overall_status == "insufficient" -> predicted "recapture"

Validation heuristic (if present):
  final_decision == "pass"         -> predicted "accept"
  final_decision == "review"       -> predicted "review"
  final_decision == "fail"         -> predicted "fail"

Ground truth comes from expected_action (or label if expected_action absent).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_FIXTURE_DIRS = [
    Path(__file__).parent / "fixtures" / "coverage_fixtures.json",
    Path(__file__).parent / "fixtures" / "real_captures" / "real_capture_fixtures.json",
]


# ─────────────────────────────────────────────────────────────────────────────
# Label / action normalization
# ─────────────────────────────────────────────────────────────────────────────

_RECAPTURE_LABELS = {"recapture", "bad", "noisy", "insufficient"}
_ACCEPT_LABELS = {"good"}

_RECAPTURE_ACTIONS = {"recapture", "fail"}
_ACCEPT_ACTIONS = {"accept"}
_REVIEW_ACTIONS = {"review"}


def _normalize_expected_binary(fixture: Dict[str, Any]) -> str:
    """Map fixture label/expected_action to binary class."""
    action = fixture.get("expected_action", "")
    label = fixture.get("label", "")
    if action in _ACCEPT_ACTIONS or label.lower() in _ACCEPT_LABELS:
        return "accept"
    return "recapture"


def _normalize_predicted_coverage(fixture: Dict[str, Any]) -> str:
    """Map coverage_report.overall_status to predicted binary class."""
    cov = fixture.get("coverage_report") or {}
    status = cov.get("overall_status", "insufficient")
    return "accept" if status == "sufficient" else "recapture"


def _normalize_predicted_validation(fixture: Dict[str, Any]) -> Optional[str]:
    """Map validation_report.final_decision to predicted class (if available)."""
    val = fixture.get("validation_report")
    if val is None:
        return None
    dec = val.get("final_decision", "")
    if dec == "pass":
        return "accept"
    if dec == "review":
        return "review"
    if dec == "fail":
        return "recapture"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-fixture evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_fixture(fixture: Dict[str, Any]) -> Dict[str, Any]:
    fid = fixture.get("fixture_id", "?")
    expected = _normalize_expected_binary(fixture)
    predicted_cov = _normalize_predicted_coverage(fixture)
    predicted_val = _normalize_predicted_validation(fixture)

    # Primary decision: coverage; fallback to validation if coverage unclear
    if predicted_val is not None and predicted_cov == "accept":
        # Both pass or val overrides
        predicted = predicted_val if predicted_val == "recapture" else predicted_cov
    else:
        predicted = predicted_cov

    # Normalize review -> accept for binary evaluation
    if predicted == "review":
        predicted = "accept"

    correct = expected == predicted

    cov = fixture.get("coverage_report") or {}
    val = fixture.get("validation_report") or {}

    return {
        "fixture_id":      fid,
        "description":     fixture.get("description", ""),
        "label":           fixture.get("label", ""),
        "expected_action": fixture.get("expected_action", ""),
        "product_type":    fixture.get("product_type", "unknown"),
        "source":          fixture.get("source", "synthetic"),
        "expected":        expected,
        "predicted":       predicted,
        "correct":         correct,
        "coverage_score":  float(cov.get("coverage_score", 0.0)),
        "unique_views":    int(cov.get("unique_views", 0)),
        "fallback_frames": int(cov.get("fallback_frames", 0)),
        "low_conf_frames": int(cov.get("low_confidence_frames", 0)),
        "cov_reasons":     cov.get("reasons", []),
        "contamination":   float(val.get("contamination_score", -1.0)),
        "val_decision":    val.get("final_decision", "n/a"),
        "issue_classes":   fixture.get("observed_issue_classes", ["none"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = tn = fp = fn = 0
    for r in results:
        e, p = r["expected"], r["predicted"]
        if e == "recapture" and p == "recapture": tp += 1
        elif e == "accept" and p == "accept":     tn += 1
        elif e == "accept" and p == "recapture":  fp += 1
        elif e == "recapture" and p == "accept":  fn += 1

    total = tp + tn + fp + fn
    acc  = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "total": total, "correct": tp + tn,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Failure breakdown
# ─────────────────────────────────────────────────────────────────────────────

def _breakdown_by_source(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List] = defaultdict(list)
    for r in results:
        groups[r["source"]].append(r)
    return {src: _compute_metrics(items) for src, items in groups.items()}


def _breakdown_by_product(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List] = defaultdict(list)
    for r in results:
        groups[r["product_type"]].append(r)
    return {pt: _compute_metrics(items) for pt, items in groups.items()}


def _breakdown_by_issue_class(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    For each issue class present in failed fixtures, count how many
    FP and FN cases involved that issue class.
    """
    fp_issues: Dict[str, int] = defaultdict(int)
    fn_issues: Dict[str, int] = defaultdict(int)

    for r in results:
        if r["correct"]:
            continue
        issues = r.get("issue_classes", ["none"])
        if r["expected"] == "accept" and r["predicted"] == "recapture":
            for iss in issues:
                fp_issues[iss] += 1
        elif r["expected"] == "recapture" and r["predicted"] == "accept":
            for iss in issues:
                fn_issues[iss] += 1

    all_issues = sorted(set(list(fp_issues.keys()) + list(fn_issues.keys())))
    return {
        iss: {"fp": fp_issues.get(iss, 0), "fn": fn_issues.get(iss, 0)}
        for iss in all_issues
    }


# ─────────────────────────────────────────────────────────────────────────────
# Threshold recommendations
# ─────────────────────────────────────────────────────────────────────────────

def _recommend_thresholds(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    issue_breakdown: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Analyses evaluation results and produces actionable threshold tuning suggestions.
    These are RECOMMENDATIONS only — no automatic rewriting.

    Returns list of recommendation dicts:
    {
        "parameter":     str,
        "config_file":   str,
        "current_value": Any,
        "direction":     "relax" | "tighten",
        "evidence":      str,
        "suggestion":    str,
    }
    """
    recs = []
    fp_list = [r for r in results if r["expected"] == "accept" and r["predicted"] == "recapture"]
    fn_list = [r for r in results if r["expected"] == "recapture" and r["predicted"] == "accept"]

    # ── FP analysis ──────────────────────────────────────────────────────────

    if issue_breakdown.get("masking_fallback", {}).get("fp", 0) > 0:
        fp_fb = issue_breakdown["masking_fallback"]["fp"]
        # Check if those FP cases had good coverage_scores
        fp_fb_scores = [
            r["coverage_score"] for r in fp_list
            if "masking_fallback" in r.get("issue_classes", [])
        ]
        avg_score = sum(fp_fb_scores) / len(fp_fb_scores) if fp_fb_scores else 0.0
        recs.append({
            "parameter":     "fallback_frames ratio threshold",
            "config_file":   "modules/capture_workflow/coverage_analyzer.py (line ~196)",
            "current_value": "readable_frames * 0.5",
            "direction":     "relax",
            "evidence":      (
                f"{fp_fb} good capture(s) rejected due to masking_fallback. "
                f"Average coverage_score of FP cases: {avg_score:.2f}. "
                "Heuristic fallback may produce acceptable masks when rembg is unavailable."
            ),
            "suggestion":    (
                "Consider raising fallback_frames ratio from 0.5 to 0.65–0.70, "
                "or checking the final contamination_score instead of the fallback_frame count alone. "
                "If validation passes (contamination_score < 0.15), fallback masking may be acceptable."
            ),
        })

    if issue_breakdown.get("low_diversity", {}).get("fp", 0) > 0:
        recs.append({
            "parameter":     "CoverageConfig.min_unique_views",
            "config_file":   "modules/capture_workflow/config.py",
            "current_value": 5,
            "direction":     "relax",
            "evidence":      (
                f"{issue_breakdown['low_diversity']['fp']} good capture(s) "
                "rejected due to low_diversity. "
                "Good final assets found despite fewer unique views."
            ),
            "suggestion":    (
                "Consider lowering min_unique_views from 5 to 4 for handheld captures. "
                "Use product_type as a filter — jewelry/small objects may need fewer angles."
            ),
        })

    if issue_breakdown.get("scale_flat", {}).get("fp", 0) > 0:
        recs.append({
            "parameter":     "CoverageConfig.min_scale_variation",
            "config_file":   "modules/capture_workflow/config.py",
            "current_value": 1.15,
            "direction":     "relax",
            "evidence":      (
                f"{issue_breakdown['scale_flat']['fp']} good capture(s) blocked "
                "due to scale_flat. Turntable captures naturally have limited z-variation."
            ),
            "suggestion":    (
                "Consider lowering min_scale_variation to 1.08 for turntable/studio environments. "
                "The span_score weights (0.15) may over-penalize when unique_views is high."
            ),
        })

    # ── FN analysis ──────────────────────────────────────────────────────────

    if issue_breakdown.get("contamination_high", {}).get("fn", 0) > 0:
        recs.append({
            "parameter":     "AssetValidator contamination threshold",
            "config_file":   "modules/asset_cleanup_pipeline/isolation.py (or validator)",
            "current_value": "~0.3 (see validator)",
            "direction":     "tighten",
            "evidence":      (
                f"{issue_breakdown['contamination_high']['fn']} contaminated capture(s) "
                "accepted by heuristic. High contamination at validation was not caught at coverage stage."
            ),
            "suggestion":    (
                "Add a soft contamination_score signal to coverage guidance "
                "when ml_segmentation_unavailable=True and fallback_frames > 30%. "
                "Or lower the contamination_score threshold in AssetValidator from ~0.3 to 0.2."
            ),
        })

    if fn_list and not recs:
        recs.append({
            "parameter":     "general recall",
            "config_file":   "modules/capture_workflow/config.py",
            "current_value": "multiple",
            "direction":     "tighten",
            "evidence":      (
                f"{len(fn_list)} bad/noisy captures passed as 'sufficient'. "
                "Reasons unclear from current dataset — more labeled examples needed."
            ),
            "suggestion":    (
                "Add more labeled negative examples to the fixture set. "
                "Review the misclassified sessions manually to identify root cause."
            ),
        })

    # ── General health ────────────────────────────────────────────────────────

    if metrics["accuracy"] >= 0.90 and not recs:
        recs.append({
            "parameter":     "general",
            "config_file":   "n/a",
            "current_value": "n/a",
            "direction":     "none",
            "evidence":      f"Accuracy={metrics['accuracy']:.2%}, F1={metrics['f1']:.2%}.",
            "suggestion":    (
                "Heuristics performing well on current dataset. "
                "Add more real-capture fixtures to increase confidence."
            ),
        })

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────────────────────────────────────

def _render_markdown(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    source_breakdown: Dict[str, Any],
    product_breakdown: Dict[str, Any],
    issue_breakdown: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    fixture_files: List[str],
) -> str:
    lines = [
        "# Unified Calibration Evaluation Report",
        "",
        f"**Fixture sources:** {', '.join(fixture_files)}",
        "",
        "## Overall Metrics",
        "| Metric | Value |",
        "|---|---|",
        f"| Total fixtures | {metrics['total']} |",
        f"| Correct | {metrics['correct']} |",
        f"| Accuracy | {metrics['accuracy']:.2%} |",
        f"| Precision | {metrics['precision']:.2%} |",
        f"| Recall | {metrics['recall']:.2%} |",
        f"| F1 | {metrics['f1']:.2%} |",
        "",
        "## Confusion Matrix",
        "| | Predicted: recapture | Predicted: accept |",
        "|---|---|---|",
        f"| **Expected: recapture** | TP = {metrics['tp']} | FN = {metrics['fn']} |",
        f"| **Expected: accept**    | FP = {metrics['fp']} | TN = {metrics['tn']} |",
        "",
    ]

    # Source breakdown
    if source_breakdown:
        lines += ["## Breakdown by Source", "| Source | Total | Accuracy | Prec | Rec | F1 |",
                  "|---|---|---|---|---|---|"]
        for src, m in sorted(source_breakdown.items()):
            lines.append(
                f"| {src} | {m['total']} | {m['accuracy']:.2%} "
                f"| {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} |"
            )
        lines.append("")

    # Product breakdown
    if product_breakdown:
        lines += ["## Breakdown by Product Type", "| Product | Total | Accuracy | FP | FN |",
                  "|---|---|---|---|---|"]
        for pt, m in sorted(product_breakdown.items()):
            lines.append(f"| {pt} | {m['total']} | {m['accuracy']:.2%} | {m['fp']} | {m['fn']} |")
        lines.append("")

    # Issue class breakdown
    if issue_breakdown:
        lines += ["## Failure Breakdown by Issue Class",
                  "| Issue Class | FP (good rejected) | FN (bad accepted) |",
                  "|---|---|---|"]
        for iss, counts in sorted(issue_breakdown.items()):
            lines.append(f"| {iss} | {counts['fp']} | {counts['fn']} |")
        lines.append("")

    # Threshold recommendations
    lines += ["## Threshold Tuning Recommendations",
              "> These are SUGGESTIONS only. No thresholds are auto-modified.", ""]
    if not recommendations:
        lines.append("No recommendations at this time.")
    else:
        for i, rec in enumerate(recommendations, 1):
            direction_label = (
                "[RELAX]" if rec["direction"] == "relax"
                else "[TIGHTEN]" if rec["direction"] == "tighten"
                else "[INFO]"
            )
            lines += [
                f"### {i}. {direction_label} `{rec['parameter']}`",
                f"**Config:** `{rec['config_file']}`  ",
                f"**Current:** `{rec['current_value']}`  ",
                f"**Direction:** {rec['direction']}",
                "",
                f"**Evidence:** {rec['evidence']}",
                "",
                f"**Suggestion:** {rec['suggestion']}",
                "",
            ]

    # Per-fixture table
    lines += ["## Per-Fixture Results",
              "| ID | Source | Product | Expected | Predicted | Score | OK? |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        ok = "[OK]" if r["correct"] else "[FAIL]"
        lines.append(
            f"| {r['fixture_id']} | {r['source']} | {r['product_type']} "
            f"| {r['expected']} | {r['predicted']} "
            f"| {r['coverage_score']:.2f} | {ok} |"
        )
    lines.append("")

    # Failure details
    failed = [r for r in results if not r["correct"]]
    lines.append("### Failure Details")
    if not failed:
        lines.append("_No misclassifications — all fixtures predicted correctly._")
    else:
        for r in failed:
            lines += [
                f"",
                f"**{r['fixture_id']}** ({r['label']} / {r['product_type']}) -> "
                f"predicted `{r['predicted']}`, expected `{r['expected']}`",
                f"  - Coverage score: {r['coverage_score']:.2f}",
                f"  - Issue classes: {r['issue_classes']}",
            ]
            for reason in r["cov_reasons"]:
                lines.append(f"  - {reason}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    fixture_paths: List[Path],
    output_path: Optional[Path] = None,
    fmt: str = "both",
) -> Dict[str, Any]:
    all_fixtures = []
    loaded_files = []
    for p in fixture_paths:
        if not p.exists():
            print(f"[WARN] Fixture file not found, skipping: {p}", file=sys.stderr)
            continue
        with open(p, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        # Support both array and single object
        if isinstance(loaded, dict):
            loaded = [loaded]
        all_fixtures.extend(loaded)
        loaded_files.append(p.name)
        print(f"[INFO] Loaded {len(loaded)} fixtures from {p.name}", file=sys.stderr)

    if not all_fixtures:
        print("[ERROR] No fixtures loaded.", file=sys.stderr)
        sys.exit(1)

    results = [_evaluate_fixture(f) for f in all_fixtures]
    metrics = _compute_metrics(results)
    source_breakdown = _breakdown_by_source(results)
    product_breakdown = _breakdown_by_product(results)
    issue_breakdown = _breakdown_by_issue_class(results)
    recommendations = _recommend_thresholds(results, metrics, issue_breakdown)

    report = {
        "fixture_files":     loaded_files,
        "total_fixtures":    len(all_fixtures),
        "metrics":           metrics,
        "source_breakdown":  source_breakdown,
        "product_breakdown": product_breakdown,
        "issue_breakdown":   issue_breakdown,
        "recommendations":   recommendations,
        "results":           results,
    }

    md = _render_markdown(
        results, metrics, source_breakdown, product_breakdown,
        issue_breakdown, recommendations, loaded_files,
    )

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
        print(f"\n[INFO] Report saved to: {output_path}", file=sys.stderr)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Unified calibration evaluator — synthetic + real fixtures."
    )
    parser.add_argument(
        "--fixtures", nargs="*", type=Path, default=None,
        help="Fixture JSON files to evaluate (default: bundled synthetic + real)",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Save report to this path (.json or .md)")
    parser.add_argument("--format", dest="fmt",
                        choices=["json", "markdown", "both"], default="both",
                        help="Output format (default: both)")
    args = parser.parse_args()

    fixture_paths = args.fixtures or _DEFAULT_FIXTURE_DIRS
    run_evaluation(fixture_paths, output_path=args.output, fmt=args.fmt)


if __name__ == "__main__":
    main()
