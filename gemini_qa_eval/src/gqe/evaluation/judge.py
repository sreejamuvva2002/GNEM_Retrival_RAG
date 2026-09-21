"""Layer 3 - LLM-as-judge (Gemini, structured output).

Two judges:

* ``grade_answer``  - absolute, rubric-anchored, 1-5 per dimension. Use it for
  the headline table and for anything you want to trend over time.
* ``compare_answers`` - pairwise preference between two modes on the same
  question, with the order randomised to cancel position bias. Pairwise
  agreement with humans is consistently higher than absolute-score agreement,
  so this is the one to trust when two modes score close.

Guardrails that make judge scores worth reading:
  - the judge model differs from the answering model (set GEMINI_JUDGE_MODEL);
  - temperature 0 and a fixed rubric, so re-runs are stable;
  - the rubric demands evidence quotes before the score, which suppresses the
    "long answer = good answer" bias;
  - ``n_samples > 1`` takes the median of repeated grades to expose instability;
  - every judge score is validated against human labels (see manual/agreement.py).
    An unvalidated judge is an opinion, not a metric.
"""

from __future__ import annotations

import json
import random
import statistics
from typing import Any

from ..config import Settings, settings as default_settings
from ..gemini_client import GeminiClient

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string", "description": "Quotes from the answer and the reference that drive the scores."},
        "correctness": {"type": "integer", "description": "1-5: factual agreement with the reference answer."},
        "completeness": {"type": "integer", "description": "1-5: how much of the reference answer's content is covered."},
        "groundedness": {"type": "integer", "description": "1-5: claims supported by the supplied context; 3 if no context was supplied."},
        "relevance": {"type": "integer", "description": "1-5: answers the question actually asked, without padding."},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}, "description": "Claims not supported by reference or context."},
        "verdict": {"type": "string", "enum": ["pass", "borderline", "fail"]},
        "rationale": {"type": "string"},
    },
    "required": ["evidence", "correctness", "completeness", "groundedness", "relevance", "unsupported_claims", "verdict", "rationale"],
}

RUBRIC = """SCORING RUBRIC (apply strictly; the default is not 3)

correctness  - 5: every factual claim agrees with the reference.
               4: minor wording or rounding differences only.
               3: core claim right, one supporting fact wrong.
               2: core claim partly wrong.
               1: core claim contradicts the reference.
completeness - 5: covers all key facts of the reference (entities, counts, attributes).
               3: covers roughly half.
               1: covers almost none.
groundedness - 5: every claim traceable to the supplied context.
               3: no context was supplied, OR context supplied and mostly followed.
               1: claims contradict or ignore the supplied context.
relevance    - 5: answers exactly what was asked, no padding.
               1: off-topic or evasive.

A correct refusal ("the records do not contain X") when the reference does
contain X is completeness 1, not a pass. A refusal when the reference also has
no answer is correctness 5.
Length is not quality. Do not reward a longer answer for being longer."""

COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "margin": {"type": "string", "enum": ["decisive", "slight", "none"]},
    },
    "required": ["reasoning", "winner", "margin"],
}

JUDGE_SYSTEM = (
    "You are a strict evaluation judge for a question-answering benchmark on the "
    "Georgia electric-vehicle supply chain. You grade only what is in front of you. "
    "You never reward verbosity, confidence, or formatting. Return JSON only."
)


def _truncate(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def grade_answer(
    *,
    question: str,
    answer: str,
    reference: str,
    context: str = "",
    client: GeminiClient,
    n_samples: int = 1,
    cfg: Settings | None = None,
) -> dict[str, Any]:
    cfg = cfg or default_settings
    prompt = f"""{RUBRIC}

QUESTION
{question}

REFERENCE ANSWER (human-validated ground truth)
{_truncate(reference, 6000) or "(none supplied - score correctness and completeness against your own knowledge and mark the rationale accordingly)"}

CONTEXT SUPPLIED TO THE MODEL
{_truncate(context, 12000) or "(none - this answer came from the model's own knowledge; groundedness must be 3)"}

ANSWER UNDER EVALUATION
{_truncate(answer, 8000) or "(empty answer)"}

Quote your evidence first, then score."""

    grades: list[dict[str, Any]] = []
    for sample in range(max(1, n_samples)):
        payload = client.generate_json(
            prompt,
            schema=GRADE_SCHEMA,
            model=cfg.judge_model,
            system=JUDGE_SYSTEM,
            temperature=0.0 if sample == 0 else 0.3,
            use_cache=(sample == 0),
        )
        if payload:
            grades.append(payload)

    if not grades:
        return _empty_grade()
    return _merge(grades)


def _merge(grades: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for field in ("correctness", "completeness", "groundedness", "relevance"):
        values = [float(g.get(field, 0) or 0) for g in grades if g.get(field) is not None]
        merged[f"judge_{field}"] = statistics.median(values) if values else float("nan")
        if len(values) > 1:
            merged[f"judge_{field}_spread"] = max(values) - min(values)
    verdicts = [str(g.get("verdict", "")) for g in grades]
    merged["judge_verdict"] = max(set(verdicts), key=verdicts.count) if verdicts else ""
    merged["judge_unsupported_claims"] = grades[0].get("unsupported_claims", []) or []
    merged["judge_n_unsupported"] = float(len(merged["judge_unsupported_claims"]))
    merged["judge_rationale"] = grades[0].get("rationale", "")
    merged["judge_evidence"] = grades[0].get("evidence", "")
    scores = [merged[f"judge_{f}"] for f in ("correctness", "completeness", "groundedness", "relevance")]
    valid = [s for s in scores if s == s]  # drop NaN
    merged["judge_overall"] = sum(valid) / len(valid) if valid else float("nan")
    return merged


def _empty_grade() -> dict[str, Any]:
    return {
        "judge_correctness": float("nan"),
        "judge_completeness": float("nan"),
        "judge_groundedness": float("nan"),
        "judge_relevance": float("nan"),
        "judge_overall": float("nan"),
        "judge_verdict": "",
        "judge_unsupported_claims": [],
        "judge_n_unsupported": float("nan"),
        "judge_rationale": "judge call failed",
        "judge_evidence": "",
    }


def compare_answers(
    *,
    question: str,
    reference: str,
    answer_a: str,
    answer_b: str,
    label_a: str,
    label_b: str,
    client: GeminiClient,
    cfg: Settings | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Blind pairwise preference. Presentation order is randomised per call."""
    cfg = cfg or default_settings
    rng = rng or random.Random(f"{question}{label_a}{label_b}")
    flipped = rng.random() < 0.5
    first, second = (answer_b, answer_a) if flipped else (answer_a, answer_b)

    prompt = f"""Two systems answered the same question. Decide which answer is better,
judged against the reference answer: factual agreement first, coverage second,
relevance third. Ignore length, tone and formatting. "tie" is a legitimate
answer when neither is clearly better.

QUESTION
{question}

REFERENCE ANSWER
{_truncate(reference, 6000)}

ANSWER A
{_truncate(first, 6000)}

ANSWER B
{_truncate(second, 6000)}"""

    payload = client.generate_json(prompt, schema=COMPARE_SCHEMA, model=cfg.judge_model, system=JUDGE_SYSTEM, temperature=0.0)
    raw_winner = str(payload.get("winner", "tie"))
    if raw_winner == "tie" or raw_winner not in ("A", "B"):
        winner = "tie"
    else:
        shown_first = label_b if flipped else label_a
        shown_second = label_a if flipped else label_b
        winner = shown_first if raw_winner == "A" else shown_second
    return {
        "question": question,
        "mode_a": label_a,
        "mode_b": label_b,
        "winner": winner,
        "margin": payload.get("margin", "none"),
        "reasoning": payload.get("reasoning", ""),
        "order_flipped": flipped,
    }


def dump_schema() -> str:
    return json.dumps(GRADE_SCHEMA, indent=2)
