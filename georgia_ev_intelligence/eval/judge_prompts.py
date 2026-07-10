"""Prompts for the LLM-as-judge answer scorer (mirrors route_generation/routing/prompts.py)."""
from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are a strict grading judge for a Georgia EV supply-chain question-answering system.

You will be given a user question, the retrieved evidence JSON that was available to the
answer-generating system, and the answer it produced.

Score two dimensions on a 1-5 integer scale:
- faithfulness: does EVERY factual claim in the answer trace back to the evidence JSON?
  5 = fully grounded, no invented facts. 1 = mostly invented or contradicts the evidence.
  An answer that says the evidence is insufficient, when the evidence truly is empty or
  irrelevant, is faithful (score high).
- relevance: does the answer actually address what the question asked?
  5 = fully addresses it. 1 = off-topic or non-responsive.

Do not reward fluent writing. Do not penalize brevity. Judge only grounding and relevance.

Return ONLY a single JSON object, no prose:
{"faithfulness": 1-5, "relevance": 1-5, "notes": "one short sentence"}"""


def build_user_prompt(question: str, evidence: dict[str, Any], answer: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        "Evidence JSON (everything the system was allowed to use):\n"
        f"{json.dumps(evidence, ensure_ascii=False, indent=2, default=str)}\n\n"
        f"Answer to grade:\n{answer}\n\n"
        "Score faithfulness and relevance. Return only the JSON object."
    )
