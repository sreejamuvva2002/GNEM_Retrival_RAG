"""Official RAGAS evaluation wired to a Kimi 2.6 OpenAI-compatible judge.

Per-row metrics:
  faithfulness       (skipped for no_rag — context is empty)
  answer_relevancy   (always)
  context_precision  (skipped for no_rag)
  context_recall     (skipped for no_rag)
  answer_correctness (always)

Final score weights (renormalized when some metrics are skipped):
  faithfulness        0.25
  answer_relevancy    0.20
  context_precision   0.20
  context_recall      0.20
  answer_correctness  0.15
"""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from llm_comparison.config import JudgeConfig

logger = logging.getLogger("llm_comparison.evaluation")

WEIGHTS: dict[str, float] = {
    "faithfulness": 0.25,
    "answer_relevancy": 0.20,
    "context_precision": 0.20,
    "context_recall": 0.20,
    "answer_correctness": 0.15,
}

ALL_METRICS = list(WEIGHTS.keys())
NO_RAG_SKIP = {"faithfulness", "context_precision", "context_recall"}


def _split_contexts(raw: Any) -> list[str]:
    """Split a stored retrieved_context cell into a list[str] for ragas.

    Pandas returns NaN for empty cells; `str(NaN) == "nan"`, so guarding only
    against `None` is not enough. We must explicitly check pd.isna() and the
    literal "nan" sentinel before treating the value as content.
    """
    if raw is None:
        return []
    try:
        if pd.isna(raw):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks or [text]


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


# ── Lazy imports so the module loads even without ragas/langchain_openai ──

def _build_judge_and_embeddings(cfg: JudgeConfig):
    from langchain_openai import ChatOpenAI

    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    chat = ChatOpenAI(
        model=cfg.judge_model,
        base_url=cfg.judge_base_url,
        api_key=cfg.judge_api_key,
        temperature=0.0,
        timeout=120,
    )
    emb = OllamaEmbeddings(model=cfg.ragas_embedding_model, base_url=cfg.ollama_base_url)
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(emb)


def _build_metrics(judge_llm, judge_embeddings) -> dict[str, Any]:
    """Instantiate the official ragas metrics, keyed by canonical name."""
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerSimilarity,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    answer_similarity = AnswerSimilarity(embeddings=judge_embeddings)
    return {
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_relevancy": ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        "context_precision": LLMContextPrecisionWithReference(llm=judge_llm),
        "context_recall": LLMContextRecall(llm=judge_llm),
        "answer_correctness": AnswerCorrectness(
            llm=judge_llm,
            embeddings=judge_embeddings,
            answer_similarity=answer_similarity,
        ),
    }


def _make_sample(question: str, answer: str, contexts: list[str], reference: str):
    from ragas.dataset_schema import SingleTurnSample

    return SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference,
    )


def _score_one(metric: Any, sample: Any) -> float:
    """Run a single ragas metric synchronously, returning a float in [0,1]."""
    if hasattr(metric, "single_turn_score"):
        return float(metric.single_turn_score(sample))
    # Fallback for async-only metrics
    import asyncio

    return float(asyncio.run(metric.single_turn_ascore(sample)))


def _final_score(scores: dict[str, float], skipped: set[str]) -> float:
    """Weighted average, renormalizing over the metrics that actually ran."""
    used = {m: w for m, w in WEIGHTS.items() if m not in skipped}
    total_weight = sum(used.values()) or 1.0
    return round(
        sum(scores[m] * (w / total_weight) for m, w in used.items()),
        4,
    )


def evaluate_rows(
    gen_df: pd.DataFrame,
    cfg: JudgeConfig,
) -> pd.DataFrame:
    """Score every generation row. Returns a per_row DataFrame."""
    judge_llm, judge_embeddings = _build_judge_and_embeddings(cfg)
    metrics = _build_metrics(judge_llm, judge_embeddings)

    out_rows: list[dict[str, Any]] = []
    total = len(gen_df)
    for i, row in enumerate(gen_df.to_dict(orient="records"), start=1):
        err = row.get("error", "")
        if not _is_blank(err):
            logger.warning(
                "Skipping row %d (model=%s mode=%s qid=%s): generation error=%s",
                i, row.get("model"), row.get("mode"), row.get("question_id"), err,
            )
            out_rows.append({**_blank_eval_row(row), "notes": f"skipped: {err}"})
            continue

        def _safe_str(value: Any) -> str:
            return "" if _is_blank(value) else str(value)

        question = _safe_str(row.get("question"))
        answer = _safe_str(row.get("answer"))
        reference = _safe_str(row.get("golden_answer"))
        contexts = _split_contexts(row.get("retrieved_context"))
        mode = _safe_str(row.get("mode"))

        scores: dict[str, float] = {m: 0.0 for m in ALL_METRICS}
        skipped: set[str] = set()
        notes_parts: list[str] = []

        # Only no_rag mode legitimately runs without contexts. A RAG mode
        # with zero contexts is a retrieval failure that must hurt the
        # score rather than be hidden by weight renormalisation.
        if mode == "no_rag":
            skipped |= NO_RAG_SKIP
            notes_parts.append("no_rag: no retrieved context")
        elif not contexts:
            notes_parts.append("retrieval failure: 0 contexts in RAG mode")

        sample = _make_sample(question, answer, contexts, reference)

        eval_start = time.monotonic()
        for metric_name in ALL_METRICS:
            if metric_name in skipped:
                continue
            if not contexts and metric_name in NO_RAG_SKIP:
                scores[metric_name] = 0.0
                continue
            try:
                scores[metric_name] = float(_score_one(metrics[metric_name], sample))
            except Exception as exc:
                logger.warning(
                    "Metric %s failed for row %d (model=%s mode=%s qid=%s): %s",
                    metric_name, i, row.get("model"), row.get("mode"), row.get("question_id"), exc,
                )
                scores[metric_name] = 0.0
                notes_parts.append(f"{metric_name}: {type(exc).__name__}")
        eval_elapsed = time.monotonic() - eval_start

        final = _final_score(scores, skipped)

        out_rows.append(
            {
                "run_id": row.get("run_id", ""),
                "question_id": row.get("question_id", ""),
                "category": row.get("category", ""),
                "question": question,
                "golden_answer": reference,
                "model": row.get("model", ""),
                "mode": mode,
                "answer": answer,
                "faithfulness": round(scores["faithfulness"], 4),
                "answer_relevancy": round(scores["answer_relevancy"], 4),
                "context_precision": round(scores["context_precision"], 4),
                "context_recall": round(scores["context_recall"], 4),
                "answer_correctness": round(scores["answer_correctness"], 4),
                "final_score": final,
                "judge_model": cfg.judge_model,
                "eval_elapsed_s": round(eval_elapsed, 3),
                "notes": "; ".join(notes_parts),
            }
        )

        logger.info(
            "[%d/%d] scored model=%s mode=%s qid=%s final=%.3f (%.1fs)",
            i, total, row.get("model"), mode, row.get("question_id"), final, eval_elapsed,
        )

    return pd.DataFrame(out_rows)


def _blank_eval_row(gen_row: dict) -> dict[str, Any]:
    return {
        "run_id": gen_row.get("run_id", ""),
        "question_id": gen_row.get("question_id", ""),
        "category": gen_row.get("category", ""),
        "question": gen_row.get("question", ""),
        "golden_answer": gen_row.get("golden_answer", ""),
        "model": gen_row.get("model", ""),
        "mode": gen_row.get("mode", ""),
        "answer": gen_row.get("answer", ""),
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "answer_correctness": 0.0,
        "final_score": 0.0,
        "judge_model": "",
        "eval_elapsed_s": 0.0,
        "notes": "",
    }


def build_aggregations(per_row: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute the four aggregation sheets demanded by the spec."""
    metric_cols = ALL_METRICS + ["final_score"]

    if per_row.empty:
        empty = pd.DataFrame(columns=["model", "mode", *metric_cols])
        return {
            "agg_model_mode": empty,
            "agg_model_mode_metric": pd.DataFrame(columns=["model", "mode", "metric", "mean", "std", "n"]),
            "agg_category_mode": pd.DataFrame(columns=["category", "mode", *metric_cols]),
        }

    agg_model_mode = (
        per_row.groupby(["model", "mode"])[metric_cols]
        .mean()
        .round(4)
        .reset_index()
    )

    long_rows: list[dict[str, Any]] = []
    for (model, mode), sub in per_row.groupby(["model", "mode"]):
        for metric in metric_cols:
            long_rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "metric": metric,
                    "mean": round(sub[metric].mean(), 4),
                    "std": round(sub[metric].std(ddof=0), 4) if len(sub) > 0 else 0.0,
                    "n": int(len(sub)),
                }
            )
    agg_model_mode_metric = pd.DataFrame(long_rows)

    agg_category_mode = (
        per_row.groupby(["category", "mode"])[metric_cols]
        .mean()
        .round(4)
        .reset_index()
    )

    return {
        "agg_model_mode": agg_model_mode,
        "agg_model_mode_metric": agg_model_mode_metric,
        "agg_category_mode": agg_category_mode,
    }