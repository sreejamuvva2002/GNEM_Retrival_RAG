"""In-repo RAGAS evaluator using a local Ollama LLM as the judge.

WHY THIS FILE EXISTS
--------------------
After ``run_baseline.py`` produces JSONL files with LLM answers and retrieved
contexts, this script computes objective RAGAS metrics using a separate judge
LLM (default: ``qwen2.5:14b``).  This separates generation from evaluation and
allows re-evaluating the same run with a different judge or new metrics.

METRICS PER PIPELINE
--------------------
Metrics are chosen based on what each pipeline provides:

  ``rag_only``        → answer_correctness, answer_relevancy, faithfulness,
                        context_precision, context_recall
                        (has retrieved contexts → all retrieval metrics apply)

  ``hybrid_rag``      → answer_correctness, answer_relevancy, faithfulness,
                        context_precision, context_recall
                        (has same retrieved contexts as rag_only → same metrics)

  ``pretrained_only`` → answer_correctness, answer_relevancy only
                        (no contexts → context metrics not meaningful)

  ``direct_kb``       → answer_correctness, answer_relevancy, faithfulness,
                        context_precision, context_recall
                        (has all 205 KB rows as contexts; context_precision and
                        context_recall will be trivially high — interpret carefully)

JUDGE LLM SETUP
---------------
Uses LangChain Ollama wrappers (``langchain_ollama`` preferred, falls back to
``langchain_community``) wrapped with RAGAS's ``LangchainLLMWrapper``.
Requires Ollama running locally with the judge model pulled.

RAGAS DATASET FORMAT
--------------------
RAGAS ``evaluate()`` expects a HuggingFace ``Dataset`` with columns:
  - ``question``      : the user question
  - ``answer``        : the LLM-generated answer
  - ``ground_truth``  : the human-validated answer
  - ``contexts``      : list of retrieved text strings (or ``[""]`` if none)

OUTPUT
------
``<run-dir>/ragas_scores.json`` with structure::
    {
      "metadata": { "run_dir": ..., "judge_model": ..., "run_date": ... },
      "by_pipeline": {
        "<pipeline>": {
          "per_question": [ { "question_id": ..., "answer_correctness": ... } ],
          "aggregate":    { "answer_correctness": { "mean": ..., "std": ..., ... } }
        }
      }
    }

USAGE
-----
    python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.evaluate_ragas \\
        --run-dir georgia_ev_intelligence/outputs/baselines/<timestamp> \\
        --judge-model qwen2.5:14b \\
        --judge-timeout 600

    # Use a smaller/faster judge model to avoid timeouts on slower hardware:
    python -m ... --judge-model qwen2.5:7b --judge-timeout 300

Reads the JSONL files produced by ``run_baseline.py`` and computes RAGAS
metrics per pipeline.  Results are written as a JSON file inside the run
directory.

Usage:
    python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.evaluate_ragas \\
        --run-dir georgia_ev_intelligence/outputs/baselines/<timestamp> \\
        --judge-model qwen2.5:14b

    # Custom embedding model and output path:
    python -m ... \\
        --run-dir ... \\
        --judge-model qwen2.5:14b \\
        --embed-model nomic-embed-text \\
        --output ragas_scores.json

Metrics computed per pipeline:
    rag_only        — answer_correctness, answer_relevancy, faithfulness,
                      context_precision, context_recall
    hybrid_rag      — answer_correctness, answer_relevancy, faithfulness
    pretrained_only — answer_correctness, answer_relevancy
    direct_kb       — answer_correctness, answer_relevancy, faithfulness,
                      context_precision, context_recall
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy imports for optional RAGAS / LangChain dependencies
# ---------------------------------------------------------------------------

def _import_ragas():
    """Import RAGAS components and return them as a namespace."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            Faithfulness,
            ContextPrecision,
            ContextRecall,
        )
        return {
            "Dataset": Dataset,
            "evaluate": evaluate,
            "LangchainLLMWrapper": LangchainLLMWrapper,
            "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
            "AnswerCorrectness": AnswerCorrectness,
            "AnswerRelevancy": AnswerRelevancy,
            "Faithfulness": Faithfulness,
            "ContextPrecision": ContextPrecision,
            "ContextRecall": ContextRecall,
        }
    except ImportError as exc:
        raise ImportError(
            "RAGAS dependencies not installed. "
            "Run: pip install ragas langchain langchain-community langchain-ollama"
        ) from exc


def _import_langchain(
    judge_model: str,
    embed_model: str,
    ollama_base_url: str,
    timeout: int = 300,
):
    """Build LangChain-wrapped Ollama LLM and embedding model."""
    try:
        from langchain_ollama import OllamaLLM, OllamaEmbeddings
        llm = OllamaLLM(model=judge_model, base_url=ollama_base_url, timeout=timeout)
        embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_base_url)
        return llm, embeddings
    except ImportError:
        pass

    # Fallback to langchain-community
    try:
        from langchain_community.llms import Ollama
        from langchain_community.embeddings import OllamaEmbeddings
        llm = Ollama(model=judge_model, base_url=ollama_base_url, timeout=timeout)
        embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_base_url)
        return llm, embeddings
    except ImportError as exc:
        raise ImportError(
            "Neither langchain-ollama nor langchain-community is installed. "
            "Run: pip install langchain-ollama"
        ) from exc


# ---------------------------------------------------------------------------
# Metrics configuration per pipeline
# ---------------------------------------------------------------------------

# Each entry: (metric_name, needs_context)
_PIPELINE_METRICS: dict[str, list[str]] = {
    "rag_only": [
        "answer_correctness",
        "answer_relevancy",
        "faithfulness",
        "context_precision",
        "context_recall",
    ],
    "hybrid_rag": [
        "answer_correctness",
        "answer_relevancy",
        "faithfulness",
        "context_precision",
        "context_recall",
    ],
    "pretrained_only": [
        "answer_correctness",
        "answer_relevancy",
    ],
    "direct_kb": [
        "answer_correctness",
        "answer_relevancy",
        "faithfulness",
        "context_precision",
        "context_recall",
    ],
}

_ALL_KNOWN_PIPELINES = set(_PIPELINE_METRICS)


# ---------------------------------------------------------------------------
# JSONL loading helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _discover_jsonl_by_pipeline(
    run_dir: Path,
    pipeline_filter: set[str] | None,
    model_filter: set[str] | None,
) -> dict[str, list[dict]]:
    """Return {pipeline_name: [records, ...]} grouped by pipeline.

    Multiple JSONL files for the same pipeline (different models) are merged —
    this lets us evaluate all questions regardless of which model produced them.
    When a question appears in multiple files for the same pipeline, the first
    occurrence wins (same order as run_baseline output).
    """
    by_pipeline: dict[str, list[dict]] = {}

    for jsonl_path in sorted(run_dir.glob("*.jsonl")):
        records = _load_jsonl(jsonl_path)
        if not records:
            continue
        pipeline = records[0].get("pipeline", "")
        model = records[0].get("model", "")
        if pipeline_filter and pipeline not in pipeline_filter:
            continue
        if model_filter and model not in model_filter:
            continue

        if pipeline not in by_pipeline:
            by_pipeline[pipeline] = []
        # Deduplicate by question — keep first seen
        existing_questions = {r["question"] for r in by_pipeline[pipeline]}
        for rec in records:
            if rec["question"] not in existing_questions:
                by_pipeline[pipeline].append(rec)
                existing_questions.add(rec["question"])

    return by_pipeline


# ---------------------------------------------------------------------------
# RAGAS dataset builders
# ---------------------------------------------------------------------------

def _build_ragas_dataset(records: list[dict], pipeline: str, ragas_ns: dict):
    """Convert JSONL records to a ragas ``Dataset``."""
    Dataset = ragas_ns["Dataset"]

    questions: list[str] = []
    answers: list[str] = []
    ground_truths: list[str] = []
    contexts_list: list[list[str]] = []

    for rec in records:
        questions.append(rec["question"])
        answers.append(rec.get("answer") or "")
        ground_truths.append(rec.get("ground_truth") or "")

        raw_ctx = rec.get("contexts") or []
        if pipeline == "pretrained_only" or not raw_ctx:
            # RAGAS requires at least one non-empty string per row
            contexts_list.append([""])
        else:
            contexts_list.append([str(c) for c in raw_ctx if c])

    data = {
        "question": questions,
        "answer": answers,
        "ground_truth": ground_truths,
        "contexts": contexts_list,
    }
    return Dataset.from_dict(data)


def _build_metrics(metric_names: list[str], ragas_ns: dict, ragas_llm, ragas_embed):
    """Instantiate requested RAGAS metric objects."""
    metric_map = {
        "answer_correctness": ragas_ns["AnswerCorrectness"],
        "answer_relevancy": ragas_ns["AnswerRelevancy"],
        "faithfulness": ragas_ns["Faithfulness"],
        "context_precision": ragas_ns["ContextPrecision"],
        "context_recall": ragas_ns["ContextRecall"],
    }
    metrics = []
    for name in metric_names:
        cls = metric_map[name]
        try:
            m = cls(llm=ragas_llm, embeddings=ragas_embed)
        except TypeError:
            # Some metrics don't accept embeddings
            try:
                m = cls(llm=ragas_llm)
            except TypeError:
                m = cls()
        metrics.append(m)
    return metrics


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def _aggregate(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    pipeline: str,
    records: list[dict],
    ragas_ns: dict,
    ragas_llm,
    ragas_embed,
) -> dict:
    """Evaluate one pipeline and return per-question + aggregate scores."""
    print(f"  Evaluating pipeline '{pipeline}' ({len(records)} questions) ...")
    metric_names = _PIPELINE_METRICS.get(pipeline, ["answer_correctness", "answer_relevancy"])

    dataset = _build_ragas_dataset(records, pipeline, ragas_ns)
    metrics = _build_metrics(metric_names, ragas_ns, ragas_llm, ragas_embed)

    evaluate_fn = ragas_ns["evaluate"]
    result = evaluate_fn(dataset=dataset, metrics=metrics)

    # Convert to pandas then to list-of-dicts for JSON serialisation
    result_df = result.to_pandas()

    per_question: list[dict] = []
    for idx, row in result_df.iterrows():
        question_id = records[idx].get("question_id", idx + 1) if idx < len(records) else idx + 1
        entry: dict = {"question_id": question_id, "question": records[idx]["question"]}
        for name in metric_names:
            val = row.get(name)
            entry[name] = round(float(val), 4) if val is not None and val == val else None
        per_question.append(entry)

    aggregate: dict[str, dict] = {}
    for name in metric_names:
        values = [e[name] for e in per_question if e.get(name) is not None]
        aggregate[name] = _aggregate(values)

    return {"per_question": per_question, "aggregate": aggregate}


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    pipeline_filter = set(args.pipelines) if args.pipelines else None
    model_filter = set(args.models) if args.models else None

    by_pipeline = _discover_jsonl_by_pipeline(run_dir, pipeline_filter, model_filter)
    if not by_pipeline:
        print("No matching JSONL files found after filtering.")
        return 1

    print(f"Pipelines to evaluate: {', '.join(sorted(by_pipeline))}")
    print(f"Judge LLM     : {args.judge_model}")
    print(f"Embed model   : {args.embed_model}")
    print(f"Ollama URL    : {args.ollama_url}")

    # Lazy imports
    print("\nLoading RAGAS + LangChain ...")
    ragas_ns = _import_ragas()
    raw_llm, raw_embed = _import_langchain(
        judge_model=args.judge_model,
        embed_model=args.embed_model,
        ollama_base_url=args.ollama_url,
        timeout=args.judge_timeout,
    )
    ragas_llm = ragas_ns["LangchainLLMWrapper"](raw_llm)
    ragas_embed = ragas_ns["LangchainEmbeddingsWrapper"](raw_embed)

    # Evaluate
    results_by_pipeline: dict[str, dict] = {}
    for pipeline_name in sorted(by_pipeline):
        records = by_pipeline[pipeline_name]
        try:
            results_by_pipeline[pipeline_name] = evaluate_pipeline(
                pipeline=pipeline_name,
                records=records,
                ragas_ns=ragas_ns,
                ragas_llm=ragas_llm,
                ragas_embed=ragas_embed,
            )
        except Exception as exc:
            print(f"  ERROR evaluating '{pipeline_name}': {exc}")
            results_by_pipeline[pipeline_name] = {"error": str(exc)}

    # Write output
    output_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "ragas_scores.json"
    )
    output_data = {
        "metadata": {
            "run_dir": str(run_dir),
            "judge_model": args.judge_model,
            "embed_model": args.embed_model,
            "ollama_url": args.ollama_url,
            "run_date": datetime.now(tz=timezone.utc).isoformat(),
        },
        "by_pipeline": results_by_pipeline,
    }
    output_path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nRAGAS scores saved to: {output_path}")

    # Print a quick summary table
    _print_summary(results_by_pipeline)
    return 0


def _print_summary(results_by_pipeline: dict[str, dict]) -> None:
    """Print a compact per-pipeline aggregate summary to stdout."""
    print("\n" + "=" * 72)
    print("RAGAS Summary (means)")
    print("=" * 72)
    for pipeline, results in sorted(results_by_pipeline.items()):
        if "error" in results:
            print(f"  {pipeline}: ERROR — {results['error']}")
            continue
        agg = results.get("aggregate", {})
        row_parts = [f"{pipeline}:"]
        for metric, stats in agg.items():
            mean_val = stats.get("mean")
            row_parts.append(f"{metric}={mean_val:.3f}" if mean_val is not None else f"{metric}=n/a")
        print("  " + "  ".join(row_parts))
    print("=" * 72)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a baseline JSONL run folder with RAGAS "
            "using an Ollama LLM as the judge."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the timestamped baseline run folder (contains *.jsonl files).",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen2.5:14b",
        help="Ollama model to use as the RAGAS judge LLM (default: qwen2.5:14b).",
    )
    parser.add_argument(
        "--embed-model",
        default="nomic-embed-text",
        help="Ollama embedding model for answer_relevancy (default: nomic-embed-text).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <run-dir>/ragas_scores.json.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=None,
        metavar="PIPELINE",
        help="Evaluate only these pipelines (e.g. rag_only hybrid_rag).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Use only JSONL files from these models (e.g. gemma3:27b).",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=300,
        help=(
            "Timeout in seconds per judge LLM call (default: 300). "
            "Increase to 600 for large judge models like qwen2.5:14b on slow hardware."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
