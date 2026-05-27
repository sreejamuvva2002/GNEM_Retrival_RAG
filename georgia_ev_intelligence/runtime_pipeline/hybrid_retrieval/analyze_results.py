"""Analyse pipeline evaluation results and produce a Markdown report + CSVs.

WHY THIS FILE EXISTS
--------------------
After ``evaluate_ragas.py`` scores the pipeline outputs, this script reads
those scores and produces human-readable analysis: pipeline rankings, head-to-head
comparisons, retrieval lift measurements, and struggling-question detection.  It
is intentionally separated from the evaluator so it can be re-run quickly with
different thresholds or comparisons without re-running the expensive LLM-based
RAGAS evaluation.

INPUT FORMATS (auto-detected)
------------------------------
  1. ``ragas_scores.json``       — produced by ``evaluate_ragas.py`` (preferred)
                                   Metrics: answer_correctness, faithfulness,
                                            context_precision, context_recall,
                                            answer_relevancy
  2. ``ragas_report_ragas.xlsx`` — produced by the legacy custom RAGAS scorer
                                   Metrics: answer_accuracy, faithfulness,
                                            response_groundedness,
                                            answer_relevancy, composite_score

OUTPUTS
-------
All written to the run directory (or ``--output-dir``):
  ``analysis.md``              — full 6-section Markdown report
  ``per_question_scores.csv``  — one row per (question × pipeline), all metrics
  ``pipeline_summary.csv``     — aggregate mean ± std per pipeline × metric

REPORT SECTIONS
---------------
  1. Aggregate scores per pipeline (mean ± std, best value bolded)
  2. Pipeline ranking by primary metric
  3. Head-to-head: rag_only vs hybrid_rag (+ direct_kb)
  4. Retrieval lift: rag_only vs pretrained_only
  5. Struggling questions (all retrieval pipelines scored < threshold)
  6. Full metric detail per pipeline (mean, std, min, max, N)

KEY ANALYSIS FUNCTIONS
-----------------------
``_head_to_head(scores, a, b, metric, threshold=0.05)``
    Compares pipelines a and b per question.  A win is Δ > 0.05 (not
    just any positive delta) to focus on meaningful differences.

``_struggling_questions(scores, pipelines, metric, threshold=0.3)``
    Finds questions where ALL retrieval pipelines scored below threshold —
    indicating a fundamental retrieval or KB coverage gap.

``_pipeline_scores_by_question(scores, pipeline, metric)``
    Normalises question string whitespace and averages across multiple
    model runs for the same pipeline (JSONL from multi-model runs).

USAGE
-----
    python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.analyze_results \\
        --run-dir georgia_ev_intelligence/outputs/baselines/<timestamp>

    python -m ... --run-dir outputs/unused/baselines/20260520_200016   # xlsx format

    The script auto-detects which format is present in the run directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class QuestionScore(NamedTuple):
    question_id: object
    question: str
    pipeline: str
    model: str
    scores: dict[str, float | None]   # {metric_name: value_or_None}


class PipelineSummary(NamedTuple):
    pipeline: str
    model: str
    aggregate: dict[str, dict]        # {metric: {mean, std, min, max, n}}


# ---------------------------------------------------------------------------
# Format 1: ragas_scores.json  (from evaluate_ragas.py)
# ---------------------------------------------------------------------------

def _load_ragas_json(path: Path) -> tuple[list[QuestionScore], dict]:
    """Load ragas_scores.json → (question_scores, metadata)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    by_pipeline: dict = data.get("by_pipeline", {})

    scores: list[QuestionScore] = []
    for pipeline, results in by_pipeline.items():
        if "error" in results:
            continue
        model = metadata.get("run_dir", "").split("/")[-1]   # fallback label
        for entry in results.get("per_question", []):
            metric_values = {
                k: v
                for k, v in entry.items()
                if k not in ("question_id", "question")
            }
            scores.append(QuestionScore(
                question_id=entry.get("question_id"),
                question=entry.get("question", ""),
                pipeline=pipeline,
                model=model,
                scores=metric_values,
            ))
    return scores, metadata


# ---------------------------------------------------------------------------
# Format 2: ragas_report_ragas.xlsx  (from custom scorer)
# ---------------------------------------------------------------------------

def _load_ragas_xlsx(path: Path) -> tuple[list[QuestionScore], dict]:
    """Load ragas_report_ragas.xlsx → (question_scores, metadata)."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to read xlsx files") from exc

    xl = pd.ExcelFile(path)
    if "ragas_scores_long" not in xl.sheet_names:
        raise ValueError(
            f"Expected sheet 'ragas_scores_long' in {path.name}. "
            f"Available: {xl.sheet_names}"
        )

    df = pd.read_excel(xl, sheet_name="ragas_scores_long")
    metric_cols = [
        c for c in df.columns
        if c not in ("question", "model", "response", "reference_answer", "context_count")
    ]

    scores: list[QuestionScore] = []
    for _, row in df.iterrows():
        raw_model = str(row["model"])
        # model column is like "gemma3_27b__rag_only" → split on last __
        parts = raw_model.rsplit("__", 1)
        model_name = parts[0] if len(parts) == 2 else raw_model
        pipeline = parts[1] if len(parts) == 2 else "unknown"

        metric_values: dict[str, float | None] = {}
        for col in metric_cols:
            val = row[col]
            if val != val or val is None:  # NaN check
                metric_values[col] = None
            else:
                metric_values[col] = float(val)

        import re as _re
        q_text = _re.sub(r"\s+", " ", str(row["question"]).strip())
        scores.append(QuestionScore(
            question_id=None,
            question=q_text,
            pipeline=pipeline,
            model=model_name,
            scores=metric_values,
        ))

    # Try to read run_info for metadata
    metadata: dict = {}
    if "ragas_run_info" in xl.sheet_names:
        ri = pd.read_excel(xl, sheet_name="ragas_run_info").to_dict(orient="records")
        if ri:
            metadata = ri[0]

    return scores, metadata


# ---------------------------------------------------------------------------
# Auto-detect format and load
# ---------------------------------------------------------------------------

def load_scores(run_dir: Path) -> tuple[list[QuestionScore], dict, str]:
    """Return (scores, metadata, source_format)."""
    json_path = run_dir / "ragas_scores.json"
    xlsx_path = run_dir / "ragas_report_ragas.xlsx"

    if json_path.exists():
        scores, meta = _load_ragas_json(json_path)
        return scores, meta, "ragas_scores.json"
    if xlsx_path.exists():
        scores, meta = _load_ragas_xlsx(xlsx_path)
        return scores, meta, "ragas_report_ragas.xlsx"

    raise FileNotFoundError(
        f"No evaluation results found in {run_dir}.\n"
        f"Expected one of:\n"
        f"  {json_path}\n"
        f"  {xlsx_path}\n"
        f"Run evaluate_ragas.py first."
    )


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def _agg(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


def _fmt(val: float | None, digits: int = 3) -> str:
    if val is None:
        return "n/a"
    return f"{val:.{digits}f}"


def build_summaries(
    scores: list[QuestionScore],
) -> dict[str, dict[str, dict]]:
    """Return {pipeline: {metric: agg_dict}} for each pipeline found."""
    from collections import defaultdict
    bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for qs in scores:
        for metric, val in qs.scores.items():
            if val is not None:
                bucket[qs.pipeline][metric].append(val)

    return {
        pipeline: {metric: _agg(vals) for metric, vals in metrics.items()}
        for pipeline, metrics in bucket.items()
    }


# ---------------------------------------------------------------------------
# Head-to-head comparison helpers
# ---------------------------------------------------------------------------

_CORRECTNESS_METRICS = (
    "answer_correctness",   # ragas_scores.json
    "answer_accuracy",      # ragas_report_ragas.xlsx
    "composite_score",      # fallback
)


def _primary_metric(scores: list[QuestionScore]) -> str:
    """Return the best available correctness metric across all records."""
    all_metrics: set[str] = set()
    for qs in scores:
        all_metrics.update(qs.scores.keys())
    for candidate in _CORRECTNESS_METRICS:
        if candidate in all_metrics:
            return candidate
    # Fall back to first available metric
    return next(iter(all_metrics)) if all_metrics else "score"


def _pipeline_scores_by_question(
    scores: list[QuestionScore],
    pipeline: str,
    metric: str,
) -> dict[str, float | None]:
    """Return {question: score} for a specific pipeline + metric.

    Question strings are normalised (strip + collapse whitespace) so that
    minor formatting differences between xlsx rows don't break the match.
    When a question appears multiple times (multiple models for the same
    pipeline), the value is averaged across non-None occurrences.
    """
    import re
    bucket: dict[str, list[float]] = {}
    for qs in scores:
        if qs.pipeline == pipeline:
            key = re.sub(r"\s+", " ", qs.question.strip())
            val = qs.scores.get(metric)
            if val is not None:
                bucket.setdefault(key, []).append(val)

    return {
        q: round(sum(vals) / len(vals), 4)
        for q, vals in bucket.items()
        if vals
    }


def _head_to_head(
    scores: list[QuestionScore],
    pipeline_a: str,
    pipeline_b: str,
    metric: str,
    threshold: float = 0.05,
) -> dict:
    """Compare two pipelines on a metric; return win/loss/tie breakdown."""
    a_scores = _pipeline_scores_by_question(scores, pipeline_a, metric)
    b_scores = _pipeline_scores_by_question(scores, pipeline_b, metric)
    common_qs = set(a_scores) & set(b_scores)

    a_wins, b_wins, ties = [], [], []
    for q in sorted(common_qs):
        a_val = a_scores[q]
        b_val = b_scores[q]
        if a_val is None or b_val is None:
            continue
        delta = a_val - b_val
        if delta > threshold:
            a_wins.append((q, a_val, b_val, delta))
        elif delta < -threshold:
            b_wins.append((q, a_val, b_val, -delta))
        else:
            ties.append((q, a_val, b_val))

    # Sort wins by delta descending
    a_wins.sort(key=lambda x: x[3], reverse=True)
    b_wins.sort(key=lambda x: x[3], reverse=True)

    return {
        "pipeline_a": pipeline_a,
        "pipeline_b": pipeline_b,
        "metric": metric,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "total": len(a_wins) + len(b_wins) + len(ties),
    }


# ---------------------------------------------------------------------------
# Struggling questions
# ---------------------------------------------------------------------------

def _struggling_questions(
    scores: list[QuestionScore],
    retrieval_pipelines: list[str],
    metric: str,
    threshold: float,
) -> list[dict]:
    """Questions where ALL retrieval-based pipelines scored below threshold."""
    q_scores: dict[str, dict[str, float | None]] = {}
    for qs in scores:
        if qs.pipeline in retrieval_pipelines:
            q_scores.setdefault(qs.question, {})[qs.pipeline] = qs.scores.get(metric)

    struggling = []
    for question, pipe_vals in q_scores.items():
        # Only include if we have data for at least 2 retrieval pipelines
        valid = {p: v for p, v in pipe_vals.items() if v is not None}
        if len(valid) < 2:
            continue
        if all(v < threshold for v in valid.values()):
            struggling.append({
                "question": question,
                "scores": pipe_vals,
                "worst": min((v for v in valid.values()), default=None),
            })

    struggling.sort(key=lambda x: x["worst"] if x["worst"] is not None else 1.0)
    return struggling


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    col_widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    data_rows = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, col_widths)) + " |"
        for row in rows
    ]
    return "\n".join([header_row, sep] + data_rows)


def build_markdown(
    scores: list[QuestionScore],
    run_dir: Path,
    source_format: str,
    metadata: dict,
    primary_metric: str,
    struggle_threshold: float,
    top_n: int,
) -> str:
    summaries = build_summaries(scores)
    pipelines = sorted(summaries.keys())
    all_metrics = sorted({m for s in summaries.values() for m in s})

    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    run_date = metadata.get("run_date", datetime.now().isoformat())[:19]
    judge = metadata.get("judge_model", "—")
    lines += [
        f"# RAGAS Analysis Report",
        f"",
        f"| Field        | Value |",
        f"|--------------|-------|",
        f"| Run directory | `{run_dir}` |",
        f"| Source file  | `{source_format}` |",
        f"| Judge LLM    | `{judge}` |",
        f"| Run date     | {run_date} |",
        f"| Report date  | {datetime.now().strftime('%Y-%m-%d %H:%M')} |",
        f"",
    ]

    # ── Section 1: Aggregate summary table ──────────────────────────────────
    lines += ["## 1. Aggregate Scores per Pipeline", ""]
    metric_headers = ["Pipeline"] + [m.replace("_", " ").title() for m in all_metrics]
    table_rows: list[list[str]] = []
    # Find winner per metric
    winners: dict[str, str] = {}
    for metric in all_metrics:
        best_mean = -1.0
        best_pipe = ""
        for pipeline in pipelines:
            agg = summaries[pipeline].get(metric, {})
            mean = agg.get("mean")
            if mean is not None and mean > best_mean:
                best_mean = mean
                best_pipe = pipeline
        winners[metric] = best_pipe

    for pipeline in pipelines:
        row = [pipeline]
        for metric in all_metrics:
            agg = summaries[pipeline].get(metric)
            if agg is None or agg.get("mean") is None:
                row.append("n/a")
            else:
                cell = f"{_fmt(agg['mean'])} ± {_fmt(agg['std'])}"
                if winners.get(metric) == pipeline:
                    cell = f"**{cell}**"
                row.append(cell)
        table_rows.append(row)

    lines.append(_md_table(metric_headers, table_rows))
    lines.append("")
    lines.append(
        "> Bold = best mean for that metric.  "
        "n/a = metric not computed for this pipeline."
    )
    lines.append("")

    # ── Section 2: Pipeline ranking by primary metric ────────────────────────
    lines += [f"## 2. Pipeline Ranking by `{primary_metric}`", ""]
    ranked = sorted(
        [
            (p, summaries[p].get(primary_metric, {}).get("mean"))
            for p in pipelines
        ],
        key=lambda x: x[1] if x[1] is not None else -1.0,
        reverse=True,
    )
    rank_rows = [
        [str(i), p, _fmt(mean), _fmt(summaries[p].get(primary_metric, {}).get("std"))]
        for i, (p, mean) in enumerate(ranked, start=1)
    ]
    lines.append(_md_table(["Rank", "Pipeline", "Mean", "Std Dev"], rank_rows))
    lines.append("")

    # ── Section 3: RAG-Only vs Hybrid-RAG head-to-head ───────────────────────
    retrieval_pairs = [
        ("rag_only", "hybrid_rag"),
        ("rag_only", "direct_kb"),
    ]
    for pipe_a, pipe_b in retrieval_pairs:
        if pipe_a not in pipelines or pipe_b not in pipelines:
            continue
        h2h = _head_to_head(scores, pipe_a, pipe_b, primary_metric)
        lines += [
            f"## 3. Head-to-Head: `{pipe_a}` vs `{pipe_b}` on `{primary_metric}`",
            "",
            f"Total questions compared: **{h2h['total']}**",
            "",
            f"| Outcome | Count | % |",
            f"|---------|-------|---|",
            f"| `{pipe_a}` wins (Δ > 0.05) | {len(h2h['a_wins'])} "
            f"| {100*len(h2h['a_wins'])//max(h2h['total'],1)}% |",
            f"| `{pipe_b}` wins (Δ > 0.05) | {len(h2h['b_wins'])} "
            f"| {100*len(h2h['b_wins'])//max(h2h['total'],1)}% |",
            f"| Tied (|Δ| ≤ 0.05) | {len(h2h['ties'])} "
            f"| {100*len(h2h['ties'])//max(h2h['total'],1)}% |",
            "",
        ]
        if h2h["b_wins"]:
            lines.append(
                f"### Top {min(top_n, len(h2h['b_wins']))} questions where "
                f"`{pipe_b}` outperforms `{pipe_a}`"
            )
            lines.append("")
            for rank, (q, a_val, b_val, delta) in enumerate(h2h["b_wins"][:top_n], 1):
                lines.append(
                    f"{rank}. **Δ = +{delta:.3f}** — "
                    f"`{pipe_b}`: {b_val:.3f} vs `{pipe_a}`: {a_val:.3f}  "
                )
                lines.append(f"   > {q[:120]}")
            lines.append("")

        if h2h["a_wins"]:
            lines.append(
                f"### Top {min(top_n, len(h2h['a_wins']))} questions where "
                f"`{pipe_a}` outperforms `{pipe_b}`"
            )
            lines.append("")
            for rank, (q, a_val, b_val, delta) in enumerate(h2h["a_wins"][:top_n], 1):
                lines.append(
                    f"{rank}. **Δ = +{delta:.3f}** — "
                    f"`{pipe_a}`: {a_val:.3f} vs `{pipe_b}`: {b_val:.3f}  "
                )
                lines.append(f"   > {q[:120]}")
            lines.append("")

    # ── Section 4: RAG lift over pretrained ──────────────────────────────────
    if "pretrained_only" in pipelines and "rag_only" in pipelines:
        lines += [
            f"## 4. Retrieval Lift: `rag_only` vs `pretrained_only`",
            f"",
            f"How often does retrieval actually help?",
            f"",
        ]
        rag_q = _pipeline_scores_by_question(scores, "rag_only", primary_metric)
        pre_q = _pipeline_scores_by_question(scores, "pretrained_only", primary_metric)
        common = set(rag_q) & set(pre_q)

        lifted, hurt, neutral = 0, 0, 0
        lift_examples, hurt_examples = [], []
        for q in common:
            r, p = rag_q.get(q), pre_q.get(q)
            if r is None or p is None:
                continue
            delta = r - p
            if delta > 0.05:
                lifted += 1
                lift_examples.append((q, r, p, delta))
            elif delta < -0.05:
                hurt += 1
                hurt_examples.append((q, r, p, -delta))
            else:
                neutral += 1

        total_cmp = lifted + hurt + neutral
        lines += [
            f"| Outcome | Count | % |",
            f"|---------|-------|---|",
            f"| RAG lifts score (Δ > 0.05)  | {lifted} | {100*lifted//max(total_cmp,1)}% |",
            f"| RAG hurts score (Δ < −0.05) | {hurt}   | {100*hurt//max(total_cmp,1)}% |",
            f"| No meaningful difference    | {neutral} | {100*neutral//max(total_cmp,1)}% |",
            f"",
        ]
        lift_examples.sort(key=lambda x: x[3], reverse=True)
        hurt_examples.sort(key=lambda x: x[3], reverse=True)
        if hurt_examples:
            lines.append(
                f"### Questions where retrieval **hurt** (pretrained_only beat rag_only)"
            )
            lines.append("")
            for q, r, p, delta in hurt_examples[:top_n]:
                lines.append(
                    f"- **Δ = −{delta:.3f}** — "
                    f"pretrained: {p:.3f}, rag_only: {r:.3f}  "
                )
                lines.append(f"  > {q[:120]}")
            lines.append("")

    # ── Section 5: Struggling questions ──────────────────────────────────────
    retrieval_pipes = [p for p in pipelines if p not in ("pretrained_only",)]
    struggling = _struggling_questions(
        scores, retrieval_pipes, primary_metric, struggle_threshold
    )
    lines += [
        f"## 5. Struggling Questions",
        f"",
        f"Questions where **all retrieval pipelines** scored below "
        f"**{struggle_threshold}** on `{primary_metric}`:  ",
        f"Found **{len(struggling)}** question(s).",
        f"",
    ]
    if struggling:
        for entry in struggling[:top_n * 2]:
            score_parts = ", ".join(
                f"`{p}`: {_fmt(v) if v is not None else 'n/a'}"
                for p, v in sorted(entry["scores"].items())
            )
            lines.append(f"- {score_parts}  ")
            lines.append(f"  > {entry['question'][:130]}")
        lines.append("")

    # ── Section 6: Per-pipeline metric distribution ──────────────────────────
    lines += ["## 6. Metric Detail per Pipeline", ""]
    for pipeline in pipelines:
        agg_data = summaries[pipeline]
        lines.append(f"### `{pipeline}`")
        lines.append("")
        detail_rows = []
        for metric in sorted(agg_data.keys()):
            a = agg_data[metric]
            detail_rows.append([
                metric.replace("_", " ").title(),
                _fmt(a.get("mean")),
                _fmt(a.get("std")),
                _fmt(a.get("min")),
                _fmt(a.get("max")),
                str(a.get("n", 0)),
            ])
        lines.append(
            _md_table(["Metric", "Mean", "Std", "Min", "Max", "N"], detail_rows)
        )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

def write_per_question_csv(scores: list[QuestionScore], path: Path) -> None:
    """One row per (question × pipeline × metric)."""
    all_metrics = sorted({m for qs in scores for m in qs.scores})
    fieldnames = ["question_id", "question", "pipeline", "model"] + all_metrics

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for qs in scores:
            row: dict = {
                "question_id": qs.question_id or "",
                "question": qs.question,
                "pipeline": qs.pipeline,
                "model": qs.model,
            }
            for m in all_metrics:
                val = qs.scores.get(m)
                row[m] = "" if val is None else f"{val:.4f}"
            writer.writerow(row)
    print(f"  Wrote per-question CSV: {path}")


def write_summary_csv(summaries: dict[str, dict[str, dict]], path: Path) -> None:
    """One row per (pipeline × metric) with mean ± std."""
    rows: list[dict] = []
    for pipeline, metrics in sorted(summaries.items()):
        for metric, agg in sorted(metrics.items()):
            rows.append({
                "pipeline": pipeline,
                "metric": metric,
                "mean": _fmt(agg.get("mean")),
                "std": _fmt(agg.get("std")),
                "min": _fmt(agg.get("min")),
                "max": _fmt(agg.get("max")),
                "n": agg.get("n", 0),
            })

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote summary CSV:       {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()

    print(f"Loading results from: {run_dir}")
    scores, metadata, source_format = load_scores(run_dir)
    print(f"  Source: {source_format}")
    print(f"  Records loaded: {len(scores)}")
    print(f"  Pipelines found: {sorted({s.pipeline for s in scores})}")

    primary_metric = _primary_metric(scores)
    print(f"  Primary metric: {primary_metric}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir

    # Markdown report
    md_path = out_dir / "analysis.md"
    md_content = build_markdown(
        scores=scores,
        run_dir=run_dir,
        source_format=source_format,
        metadata=metadata,
        primary_metric=primary_metric,
        struggle_threshold=args.struggle_threshold,
        top_n=args.top_n,
    )
    md_path.write_text(md_content, encoding="utf-8")
    print(f"  Wrote Markdown report:   {md_path}")

    # CSVs
    pq_path = out_dir / "per_question_scores.csv"
    write_per_question_csv(scores, pq_path)

    sum_path = out_dir / "pipeline_summary.csv"
    write_summary_csv(build_summaries(scores), sum_path)

    # Quick console summary
    summaries = build_summaries(scores)
    print(f"\n{'='*64}")
    print(f"{'Pipeline':<20} {'Metric':<28} {'Mean':>7} {'Std':>7}")
    print(f"{'-'*64}")
    for pipeline in sorted(summaries.keys()):
        for metric, agg in sorted(summaries[pipeline].items()):
            mean = agg.get("mean")
            std = agg.get("std")
            if mean is not None:
                print(
                    f"  {pipeline:<18} {metric:<28} "
                    f"{mean:7.3f} {std:7.3f}"
                )
    print(f"{'='*64}")
    print(f"\nFull report: {md_path}")

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse pipeline evaluation results (ragas_scores.json or "
            "ragas_report_ragas.xlsx) and produce a Markdown report + CSVs."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Path to the timestamped baseline run folder. "
            "Must contain ragas_scores.json or ragas_report_ragas.xlsx."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write outputs to. "
            "Defaults to <run-dir>."
        ),
    )
    parser.add_argument(
        "--struggle-threshold",
        type=float,
        default=0.3,
        help=(
            "Score threshold below which a question is considered 'struggling'. "
            "A question is flagged when ALL retrieval pipelines score below this. "
            "(default: 0.3)"
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help=(
            "Number of example questions to show per head-to-head section. "
            "(default: 5)"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
