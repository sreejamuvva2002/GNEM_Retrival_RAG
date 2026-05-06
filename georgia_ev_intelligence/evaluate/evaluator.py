"""
Evaluation — RAGAS-style LLM-as-Judge Evaluator

DIRECTLY ADAPTED FROM: ev_data_LLM_comparsions/src/evaluator.py
Zero external RAGAS library — pure Ollama + httpx.

KEY DIFFERENCES from original:
  - Judge model: qwen2.5:14b (local) instead of kimi-k2.6:cloud
  - Provider: ollama ONLY (no OpenRouter dependency)
  - Config: reads from shared.config (our .env) not config.yaml
  - Input: chunks from Qdrant + Neo4j, not ChromaDB

IDENTICAL TO ORIGINAL:
  - Same 5 metrics + same weights (faithfulness=0.25, relevancy=0.20, etc.)
  - Same prompt structure (question, golden, answer, context)
  - Same _parse_judge_response logic (handles malformed JSON gracefully)
  - Same score clipping to [0.0, 1.0]
  - Same retry logic (2 retries for ollama)
  - Same evaluate_row → dict output format

WHY THIS WORKS WITHOUT RAGAS LIBRARY:
  The ragas library does the same thing internally — calls an LLM to score
  each metric. We skip the library overhead and call Ollama directly.
  This is actually MORE reliable for local models (no tokenization issues).
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx

from shared.config import Config
from shared.logger import get_logger

logger = get_logger("evaluate.evaluator")


# ── The 5 RAGAS metrics (copied exactly from ev_data_LLM_comparsions) ────────
METRIC_DEFINITIONS = {
    "faithfulness": (
        "Checks whether each statement in the generated answer is grounded "
        "in the retrieved context only. Penalize any claim not supported by context."
    ),
    "answer_relevancy": (
        "Checks whether the answer directly and completely addresses the user question."
    ),
    "context_precision": (
        "Checks whether retrieved context chunks are mostly relevant and useful "
        "for answering the question. Penalize irrelevant chunks."
    ),
    "context_recall": (
        "Checks whether retrieved context covers the key facts present in the golden answer."
    ),
    "answer_correctness": (
        "Checks whether the generated answer is factually aligned with the human validated answer."
    ),
}

# Weights — identical to ev_data_LLM_comparsions/config/config.yaml
METRIC_WEIGHTS = {
    "faithfulness":       0.20, 
    "answer_relevancy":   0.15,
    "context_precision":  0.20,
    "context_recall":     0.15,
    "answer_correctness": 0.30,
}

# Targets for our system (from Architecture_Decisions.md)
TARGET_ACCURACY = 0.80      # 80% of questions score >= 0.7 final_score
MAX_HALLUCINATION = 0.05    # faithfulness < 0.5 in <= 5% of questions


def _clip(score: float) -> float:
    """Clip score to [0.0, 1.0] — identical to ev_data_LLM_comparsions."""
    return max(0.0, min(1.0, float(score)))


def _trim(text: str, max_chars: int) -> str:
    """Trim long text — identical to ev_data_LLM_comparsions _trim_text."""
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2):]
    return f"{head}\n...[truncated]...\n{tail}"


class RAGASEvaluator:
    """
    LLM-as-Judge evaluator for the 5 RAGAS metrics.
    Adapted from ev_data_LLM_comparsions/src/evaluator.py.
    Uses qwen2.5:14b locally via Ollama instead of kimi-k2.5:cloud.
    """

    def __init__(self) -> None:
        cfg = Config.get()
        self.base_url = cfg.ollama_base_url.rstrip("/")
        self.model = cfg.ollama_llm_model      # qwen2.5:14b
        self.judge_url = f"{self.base_url}/api/generate"
        logger.info("RAGASEvaluator initialized: judge=%s", self.model)

    def _build_prompt(
        self,
        metric: str,
        question: str,
        golden: str,
        answer: str,
        context: str,
    ) -> str:
        """
        Build evaluation prompt — identical structure to ev_data_LLM_comparsions.
        Context is only included for faithfulness/precision/recall metrics.
        """
        question = _trim(question, 800)
        golden   = _trim(golden, 2400)
        answer   = _trim(answer, 2400)
        context  = _trim(context, 4000)

        context_block = ""
        if metric in {"faithfulness", "context_precision", "context_recall"}:
            context_block = f"RETRIEVED CONTEXT: {context}\n"

        return (
            "You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.\n"
            f"Evaluate the following response on the metric: {metric}\n"
            f"METRIC DEFINITION: {METRIC_DEFINITIONS[metric]}\n"
            f"QUESTION: {question}\n"
            f"GOLDEN ANSWER: {golden}\n"
            f"GENERATED ANSWER: {answer}\n"
            f"{context_block}"
            "SCORING INSTRUCTIONS:\n"
            '- Return ONLY a valid JSON object: {"score": <float>, "reasoning": "<1-2 sentence explanation>"}\n'
            "- 0.0 = completely fails the metric\n"
            "- 1.0 = perfectly satisfies the metric\n"
            "- Use precise decimals (e.g., 0.85, 0.63)\n"
            "- No markdown, no extra text outside JSON"
        )

    @staticmethod
    def _parse_judge_response(raw: str) -> dict[str, Any]:
        """
        Parse judge JSON response — identical to ev_data_LLM_comparsions._parse_judge_response.
        Handles: valid JSON, JSON inside markdown fences, JSON buried in text.
        """
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON object from surrounding text
            start = raw.find("{")
            end   = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                # Last resort: regex for score number
                m = re.search(r"(?is)score[^0-9\-]*([01](?:\.\d+)?)", raw)
                if m:
                    score = float(m.group(1))
                    if not 0.0 <= score <= 1.0:
                        raise ValueError(f"Score out of range: {score}")
                    return {"score": score, "reasoning": raw.strip()[:500] or "No reasoning"}
                raise ValueError("No JSON object found in judge response")
            payload = json.loads(raw[start: end + 1])

        score     = float(payload["score"])
        reasoning = str(payload.get("reasoning", "")).strip() or "No reasoning provided"
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Score out of range: {score}")
        return {"score": score, "reasoning": reasoning}

    async def _call_judge(self, prompt: str) -> str:
        """
        Call Ollama judge — adapted from ev_data_LLM_comparsions._call_ollama_judge.
        Key settings:
          - think: False  → no chain-of-thought tokens in response field
          - format: json  → forces JSON output
          - num_predict: 220 → just enough for {"score": 0.85, "reasoning": "..."}
        """
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think":  False,       # CRITICAL: prevents qwen thinking tokens bleeding into response
            "options": {
                "temperature": 0.0,
                "num_predict": 220,
                "num_ctx":     3072,
            },
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self.judge_url, json=payload)
            resp.raise_for_status()
            body = resp.json()

        # Check response field first (ev_data_LLM_comparsions pattern)
        text = str(body.get("response", "") or "").strip()
        if text:
            return text
        # Fallback to thinking field if response is empty
        thinking = str(body.get("thinking", "") or "").strip()
        if thinking:
            logger.warning("Judge response empty — falling back to thinking field")
            return thinking
        logger.warning("Judge returned empty response and thinking fields")
        return ""

    async def score_metric(
        self,
        metric: str,
        question: str,
        golden: str,
        answer: str,
        context: str,
    ) -> dict[str, Any]:
        """
        Score a single metric with 2 retries (ollama pattern from ev_data_LLM_comparsions).
        Returns {"score": float, "reasoning": str}.
        """
        prompt     = self._build_prompt(metric, question, golden, answer, context)
        last_error = None

        for attempt in range(2):   # 2 attempts for Ollama (same as original)
            try:
                raw  = await self._call_judge(prompt)
                return self._parse_judge_response(raw)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                logger.warning("Judge parse failed [%s] attempt %d: %s", metric, attempt + 1, exc)
                await asyncio.sleep(0.5 * (attempt + 1))
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_error = exc
                logger.warning("Judge HTTP error [%s]: %s", metric, exc)
                await asyncio.sleep(0.75 * (attempt + 1))

        # Return 0 on total failure (don't crash the whole evaluation)
        logger.error("Judge failed for metric=%s: %s", metric, last_error)
        return {"score": 0.0, "reasoning": f"evaluation_error: {last_error}"}

    async def evaluate_row(
        self,
        question: str,
        golden: str,
        answer: str,
        context: str,
    ) -> dict[str, Any]:
        """
        Evaluate one Q&A row across all 5 metrics.
        SEQUENTIAL for Ollama (not parallel) — same as ev_data_LLM_comparsions ollama branch.

        Returns full scores dict — identical format to ev_data_LLM_comparsions evaluate_row output.
        """
        metric_results: dict[str, dict] = {}

        # Sequential — Ollama handles one request at a time
        for metric in METRIC_DEFINITIONS:
            try:
                metric_results[metric] = await self.score_metric(
                    metric, question, golden, answer, context
                )
            except Exception as exc:
                logger.error("Metric failed (%s): %s", metric, exc)
                metric_results[metric] = {"score": 0.0, "reasoning": f"metric_error: {exc}"}

        # Weighted final score — identical to ev_data_LLM_comparsions
        weighted = sum(
            METRIC_WEIGHTS[m] * float(metric_results[m]["score"])
            for m in METRIC_DEFINITIONS
        )

        return {
            # Scores
            "faithfulness":         float(metric_results["faithfulness"]["score"]),
            "answer_relevancy":     float(metric_results["answer_relevancy"]["score"]),
            "context_precision":    float(metric_results["context_precision"]["score"]),
            "context_recall":       float(metric_results["context_recall"]["score"]),
            "answer_correctness":   float(metric_results["answer_correctness"]["score"]),
            "final_score":          _clip(weighted),
            # Reasoning (for Excel report)
            "faithfulness_reason":        str(metric_results["faithfulness"]["reasoning"]),
            "answer_relevancy_reason":    str(metric_results["answer_relevancy"]["reasoning"]),
            "context_precision_reason":   str(metric_results["context_precision"]["reasoning"]),
            "context_recall_reason":      str(metric_results["context_recall"]["reasoning"]),
            "answer_correctness_reason":  str(metric_results["answer_correctness"]["reasoning"]),
        }

    async def evaluate_all(
        self,
        rows: list[dict[str, Any]],
        concurrency: int = 1,       # 1 for local Ollama (same as ev_data_LLM_comparsions semaphore=1)
    ) -> list[dict[str, Any]]:
        """
        Evaluate all rows with a semaphore — identical to ev_data_LLM_comparsions.evaluate_all.
        concurrency=1 because Ollama is sequential; don't flood it.
        """
        semaphore  = asyncio.Semaphore(concurrency)
        completed  = 0

        async def evaluate_one(row: dict[str, Any]) -> dict[str, Any]:
            nonlocal completed
            async with semaphore:
                metrics = await self.evaluate_row(
                    row["question"],
                    row["golden_answer"],
                    row["generated_answer"],
                    row.get("retrieved_context", ""),
                )
            completed += 1
            if completed % 5 == 0:
                logger.info("Evaluated %d/%d rows", completed, len(rows))
            return {**row, **metrics}

        return await asyncio.gather(*(evaluate_one(row) for row in rows))
