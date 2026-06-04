"""Answer generation that intentionally receives no retrieved context.

WHY THIS FILE EXISTS
--------------------
Implements the ``pretrained_only`` pipeline: the LLM answers questions about
the Georgia EV supply chain using ONLY its pretrained world knowledge — no KB
retrieval, no context injection.

PURPOSE
-------
This pipeline is the "no retrieval" lower-bound baseline.  Comparing
``rag_only`` vs ``pretrained_only`` scores reveals how much value the retrieval
system adds.  If ``rag_only`` scores are only marginally better than
``pretrained_only``, the retrieval pipeline is not contributing meaningfully.
See ``analyze_results.py`` Section 4 (Retrieval Lift) for this comparison.

PROMPT DESIGN (correctness-critical)
--------------------------------------
The prompt template has ONLY ``{user_question}`` — there is no
``{retrieved_context}`` placeholder and no context is passed.  The prompt
explicitly instructs the model to:
  - Answer using only pretrained knowledge.
  - Say what is uncertain rather than inventing specific facts.

CORRECTNESS CONTRACT
--------------------
✅ ``OnlyPretrainedAnswerPipeline.answer(question, timeout)`` takes NO
   ``retrieved_context`` argument — there is no way to accidentally pass context.
✅ In ``run_baseline.py::_answer_pretrained``, this pipeline is called as:
   ``pipeline.answer(question=row.question, timeout=self._llm_timeout)``
   — no retrieval is performed and no contexts are passed.
✅ The JSONL ``contexts`` field for this pipeline is always ``[]`` (empty list).
   RAGAS skips context-based metrics (context_precision, context_recall,
   faithfulness) for this pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from georgia_ev_intelligence.runtime_pipeline.generation.llm_client import generate_answer

from .interfaces import NoContextPromptBuilder


@dataclass(frozen=True)
class OnlyPretrainedPromptBuilder:
    """Build the no-context prompt for the Only Pre-Trained pipeline."""

    def build(self, question: str) -> str:
        return ONLY_PRETRAINED_PROMPT_TEMPLATE.format(user_question=question)


class OnlyPretrainedAnswerPipeline:
    """Generate an answer without passing any retrieved context to the LLM."""

    def __init__(
        self,
        prompt_builder: NoContextPromptBuilder | None = None,
        answer_generator: Callable[[str, int], str] = generate_answer,
    ) -> None:
        self._prompt_builder = prompt_builder or OnlyPretrainedPromptBuilder()
        self._answer_generator = answer_generator

    def answer(self, question: str, timeout: int = 180) -> str:
        prompt = self._prompt_builder.build(question=question)
        return self._answer_generator(prompt, timeout)


ONLY_PRETRAINED_PROMPT_TEMPLATE = """You are answering questions about the EV supply chain
for the state of Georgia using only what you already know from pre-training.

No retrieved knowledge-base context is provided. Answer the user question as
directly as possible. If you are uncertain, say what is uncertain rather than
inventing precise companies, products, OEMs, locations, employment numbers, or
counts.

User question:
{user_question}

Answer:"""
