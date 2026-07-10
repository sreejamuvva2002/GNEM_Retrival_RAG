"""Pydantic schema for LLM-as-judge answer scoring (mirrors route_generation/schemas.py)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerJudgment(BaseModel):
    """Structured judgment of one generated answer against its evidence."""

    faithfulness: int = Field(
        ge=1,
        le=5,
        description=(
            "Every claim in the answer is traceable to the evidence JSON. "
            "5 = fully grounded, no invented facts. 1 = largely invented or contradicts the evidence."
        ),
    )
    relevance: int = Field(
        ge=1,
        le=5,
        description=(
            "The answer actually addresses the question asked. "
            "5 = fully addresses it. 1 = off-topic or non-responsive."
        ),
    )
    notes: str = Field(default="", description="One short sentence explaining the scores.")
