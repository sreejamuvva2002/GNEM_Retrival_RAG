#!/usr/bin/env python3
"""Score routing accuracy against ``expected_route`` labels.

Reads a questions CSV (``question_id, question, expected_route[, template_id]``),
runs every question through the EXISTING route-generation pipeline -
``build_default_route_service().route_with_trace(question).final_route`` - the
same call used by ``scripts/generate_final_routes.py`` - and compares the
resulting route against ``expected_route``. Does NOT modify the router,
validator, or pre-router.

Outputs (UTF-8, overwritten each run):

  * outputs/routing_eval_results.jsonl  - one record per question (pass/fail + reason)
  * outputs/routing_eval_summary.json   - overall + per-route accuracy, confusion matrix

Run from the project root:

    python scripts/eval_routing.py
    python scripts/eval_routing.py --input data/questions_generated.csv --output-dir outputs/generated_eval

This is independent of scripts/generate_final_routes.py - it calls the router
itself, so it takes the questions CSV directly, not a final_routes.jsonl file.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from georgia_ev_intelligence.route_generation import build_default_route_service


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_questions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def evaluate(
    input_path: Path,
    results_path: Path,
    summary_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = _read_questions(input_path)
    service = build_default_route_service()

    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    confusion: dict[str, Counter] = defaultdict(Counter)
    per_route_total: Counter = Counter()
    per_route_correct: Counter = Counter()
    n_scored = n_correct = n_errors = 0

    with results_path.open("w", encoding="utf-8") as fh:
        for position, row in enumerate(rows, start=1):
            if limit is not None and position > limit:
                break

            question = (row.get("question") or "").strip()
            if not question:
                continue
            question_id = row.get("question_id") or f"g{position:03d}"
            expected = (row.get("expected_route") or "").strip() or None

            actual: str | None = None
            reason = ""
            confidence: float | None = None
            error: str | None = None
            try:
                trace = service.route_with_trace(question)
                actual = trace.final_route.route.value
                reason = trace.final_route.reason
                confidence = trace.final_route.confidence
            except Exception as exc:  # per-row isolation: never abort the batch
                error = f"{type(exc).__name__}: {exc}"
                n_errors += 1

            correct = expected is not None and actual == expected
            if expected is not None:
                per_route_total[expected] += 1
                confusion[expected][actual or "ERROR"] += 1
                n_scored += 1
                if correct:
                    per_route_correct[expected] += 1
                    n_correct += 1

            record = {
                "question_id": question_id,
                "question": question,
                "expected_route": expected,
                "actual_route": actual,
                "correct": correct,
                "confidence": confidence,
                "reason": reason,
                "error": error,
                "generated_at": _utc_now_iso(),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            status = "PASS" if correct else ("ERR " if error else "FAIL")
            print(f"[{status}] {question_id}: expected={expected} actual={actual}")

    accuracy = (n_correct / n_scored) if n_scored else None
    per_route_accuracy = {
        route: (per_route_correct[route] / total if total else None)
        for route, total in sorted(per_route_total.items())
    }
    summary = {
        "n_scored": n_scored,
        "n_correct": n_correct,
        "n_errors": n_errors,
        "accuracy": accuracy,
        "per_route_accuracy": per_route_accuracy,
        "confusion_matrix": {route: dict(counts) for route, counts in confusion.items()},
        "generated_at": _utc_now_iso(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score routing accuracy against expected_route labels.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "questions_generated.csv",
        help="CSV with columns question_id,question,expected_route (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for the JSON/JSONL outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: process only the first N rows (smoke testing).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results_path = args.output_dir / "routing_eval_results.jsonl"
    summary_path = args.output_dir / "routing_eval_summary.json"

    print(f"Input:   {args.input}")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print("-" * 48)

    try:
        summary = evaluate(
            input_path=args.input,
            results_path=results_path,
            summary_path=summary_path,
            limit=args.limit,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    print("-" * 48)
    accuracy = summary["accuracy"]
    print(f"Scored     : {summary['n_scored']}")
    print(f"Correct    : {summary['n_correct']}")
    print(f"Errors     : {summary['n_errors']}")
    print(f"Accuracy   : {accuracy:.1%}" if accuracy is not None else "Accuracy   : n/a")
    print("Per-route accuracy:")
    for route, acc in summary["per_route_accuracy"].items():
        print(f"  {route:24s}: {acc:.1%}" if acc is not None else f"  {route:24s}: n/a")
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
