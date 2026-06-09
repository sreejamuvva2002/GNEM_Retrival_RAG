#!/usr/bin/env python3
"""Execute the validated routes in ``outputs/final_routes.jsonl`` (execution README §16).

Thin batch driver around the route-execution stage. For each record it calls
``execute_route(final_route)`` and writes a uniform result row. It does NOT call
the router, modify the route, or let an LLM write SQL — all of that lives in the
``georgia_ev_intelligence.route_execution`` package.

Outputs (UTF-8 JSONL, overwritten each run):

  * outputs/execution_results.jsonl  - one record per successfully executed route
  * outputs/execution_failed.jsonl   - one record per route that errored

Run from the project root:

    python scripts/run_execution_from_routes.py
    python scripts/run_execution_from_routes.py --limit 5
    python scripts/run_execution_from_routes.py --use-llm
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from georgia_ev_intelligence.route_execution import execute_route
from georgia_ev_intelligence.route_execution.schemas import STATUS_FAILED


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(value: Any) -> Any:
    """Make DB-native types (NUMERIC -> Decimal, dates) JSON serialisable."""
    if isinstance(value, Decimal):
        # Integral decimals -> int, otherwise float, to keep counts clean.
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _read_routes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Routes file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def run(
    input_path: Path,
    results_path: Path,
    failed_path: Path,
    *,
    use_llm: bool = False,
    limit: int | None = None,
) -> tuple[int, int]:
    """Execute every route and write the JSONL outputs. Returns ``(n_ok, n_failed)``."""
    records = _read_routes(input_path)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_failed = 0

    with results_path.open("w", encoding="utf-8") as ok_fh, \
            failed_path.open("w", encoding="utf-8") as fail_fh:
        for position, record in enumerate(records, start=1):
            if limit is not None and position > limit:
                break

            question_id = record.get("question_id") or f"q{position:03d}"
            question = record.get("question") or record.get("final_route", {}).get("question", "")
            final_route = record.get("final_route") or {}
            generated_at = _utc_now_iso()

            try:
                result = execute_route(final_route, use_llm=use_llm)
                out = {
                    "question_id": question_id,
                    "question": question,
                    "status": result.status,
                    "final_route": final_route,
                    "evidence": result.evidence,
                    "answer": result.answer,
                    "generated_at": generated_at,
                }
                if result.error is not None:
                    out["error"] = result.error

                if result.status == STATUS_FAILED:
                    fail_fh.write(json.dumps(out, ensure_ascii=False, default=_json_default) + "\n")
                    n_failed += 1
                    print(f"[FAILED] {question_id} -> {final_route.get('route')}: {result.error}")
                else:
                    ok_fh.write(json.dumps(out, ensure_ascii=False, default=_json_default) + "\n")
                    n_ok += 1
                    print(f"[ok]     {question_id} -> {final_route.get('route')}")
            except Exception as exc:  # per-row isolation: never abort the batch
                out = {
                    "question_id": question_id,
                    "question": question,
                    "status": STATUS_FAILED,
                    "final_route": final_route,
                    "error": f"{type(exc).__name__}: {exc}",
                    "generated_at": generated_at,
                }
                fail_fh.write(json.dumps(out, ensure_ascii=False, default=_json_default) + "\n")
                n_failed += 1
                print(f"[FAILED] {question_id} -> {final_route.get('route')}: {out['error']}")

    return n_ok, n_failed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute validated routes from final_routes.jsonl into evidence + answers.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "final_routes.jsonl",
        help="Validated routes JSONL (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for the JSONL outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Re-ground answers via the local Ollama model (otherwise deterministic).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: execute only the first N routes (smoke testing).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results_path = args.output_dir / "execution_results.jsonl"
    failed_path = args.output_dir / "execution_failed.jsonl"

    print(f"Input:   {args.input}")
    print(f"Results: {results_path}")
    print(f"Failed:  {failed_path}")
    print(f"LLM:     {'on' if args.use_llm else 'off (deterministic)'}")
    print("-" * 48)

    try:
        n_ok, n_failed = run(
            input_path=args.input,
            results_path=results_path,
            failed_path=failed_path,
            use_llm=args.use_llm,
            limit=args.limit,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    print("-" * 48)
    print(f"Executed OK : {n_ok}")
    print(f"Failed      : {n_failed}")
    print(f"Wrote {results_path}")
    print(f"Wrote {failed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
