"""The three answer pipelines under comparison.

1. ``llm_only``  - Gemini alone, parametric knowledge, no retrieval.
2. ``llm_kb``    - Gemini + top-k rows retrieved from the internal workbook.
3. ``llm_kb_web``- the same internal context, plus Google Search grounding.

The prompts are deliberately parallel: same task framing, same output shape.
The only thing that varies is what evidence the model is given, so a score
difference between modes is attributable to the evidence and not to prompt
engineering.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import MODE_KB, MODE_KB_WEB, MODE_LLM, Settings, settings as default_settings
from .data import Question
from .gemini_client import GeminiClient
from .retrieval import Hit, Retriever, format_context

ANSWER_STYLE = (
    "Answer in plain prose or a short list. Lead with the direct answer "
    "(including the count when the question asks 'how many' or 'which'). "
    "Be specific: name companies, roles, tiers and locations rather than "
    "describing them in general terms. Do not pad the answer."
)

SYSTEM_LLM_ONLY = (
    "You are an analyst answering questions about the Georgia (USA) electric-vehicle "
    "supply chain. You have no reference documents for this task: answer from your own "
    "knowledge. If you do not know a specific fact, say so explicitly instead of guessing. "
    + ANSWER_STYLE
)

SYSTEM_KB = (
    "You are an analyst answering questions about the Georgia (USA) electric-vehicle "
    "supply chain using the company records supplied below. Use ONLY those records. "
    "Cite the record you used after each claim, as [C1], [C2], and so on. "
    "If the records do not contain the answer, say exactly what is missing rather than "
    "filling the gap from memory. " + ANSWER_STYLE
)

SYSTEM_KB_WEB = (
    "You are an analyst answering questions about the Georgia (USA) electric-vehicle "
    "supply chain. You have two evidence sources: the internal company records supplied "
    "below, and Google Search. Prefer the internal records for anything they cover, and "
    "cite them as [C1], [C2]. Use web search for what the records do not cover - recent "
    "news, announcements, figures dated after the records - and mark those claims [WEB]. "
    "If the two sources disagree, say so and give both. " + ANSWER_STYLE
)


@dataclass
class AnswerRecord:
    qid: str
    mode: str
    repeat: int
    question: str
    category: str
    golden_answer: str
    answer: str
    model: str
    contexts: list[str] = field(default_factory=list)
    context_ids: list[str] = field(default_factory=list)
    context_titles: list[str] = field(default_factory=list)
    web_sources: list[dict[str, str]] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    mocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _user_prompt(question: str, context: str | None) -> str:
    if context:
        return f"INTERNAL COMPANY RECORDS\n{context}\n\nQUESTION\n{question}\n\nANSWER"
    return f"QUESTION\n{question}\n\nANSWER"


def answer_question(
    question: Question,
    mode: str,
    *,
    client: GeminiClient,
    retriever: Retriever | None = None,
    repeat: int = 0,
    temperature: float | None = None,
    cfg: Settings | None = None,
) -> AnswerRecord:
    cfg = cfg or default_settings
    hits: list[Hit] = []
    context = None

    if mode in (MODE_KB, MODE_KB_WEB):
        if retriever is None:
            raise ValueError(f"mode {mode} needs a retriever")
        hits = retriever.search(question.question, cfg.top_k)
        context = format_context(hits)

    system = {MODE_LLM: SYSTEM_LLM_ONLY, MODE_KB: SYSTEM_KB, MODE_KB_WEB: SYSTEM_KB_WEB}[mode]

    started = time.perf_counter()
    result = client.generate(
        _user_prompt(question.question, context),
        system=system,
        temperature=temperature,
        use_web_search=(mode == MODE_KB_WEB),
        # Repeats for the self-consistency metric must not all hit the same cache entry.
        use_cache=(repeat == 0),
    )
    wall_ms = (time.perf_counter() - started) * 1000

    return AnswerRecord(
        qid=question.qid,
        mode=mode,
        repeat=repeat,
        question=question.question,
        category=question.category,
        golden_answer=question.golden_answer,
        answer=result.text,
        model=result.model,
        contexts=[f"[C{hit.rank}] {hit.doc.text}" for hit in hits],
        context_ids=[hit.doc.doc_id for hit in hits],
        context_titles=[hit.doc.title for hit in hits],
        web_sources=result.web_sources,
        search_queries=result.search_queries,
        latency_ms=result.latency_ms or wall_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        error=result.error,
        mocked=result.mocked,
    )
