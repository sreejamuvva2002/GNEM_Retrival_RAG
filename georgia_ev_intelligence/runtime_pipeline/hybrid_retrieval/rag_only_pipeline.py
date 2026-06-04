"""Answer generation that is constrained to retrieved context only.

WHY THIS FILE EXISTS
--------------------
Implements the ``rag_only`` pipeline: the strictest RAG variant where the LLM
is ONLY allowed to use the retrieved parent chunks.  No pretrained knowledge
supplementation is permitted.  This pipeline is the RAG baseline — its scores
reveal how well the retrieval system alone supports answering the questions.

PROMPT DESIGN (correctness-critical)
--------------------------------------
The prompt explicitly instructs the model to:
  1. Use ONLY the provided context.
  2. Do NOT use outside knowledge.
  3. Cite every factual claim from the context.
  4. Say "The provided knowledge base does not contain enough information..." if
     the context is insufficient — rather than hallucinating.

This strict framing is deliberate: any answer that scores well here means the
retrieval system successfully retrieved the relevant parent chunks.

CONTEXT PASSED
--------------
``retrieved_context`` is the concatenated text of up to 45 reranked parent
chunks (formatted by ``_format_retrieved_context`` in ``run_hybrid_rag.py``).
It is injected into ``{retrieved_context}`` in the prompt template.

CORRECTNESS CONTRACT
--------------------
✅ This pipeline ONLY receives retrieved context — no KB file, no empty context.
   Verified in ``run_baseline.py::_answer_rag_only``:
   ``pipeline.answer(question=..., retrieved_context=retrieval["formatted_context"])``
✅ ``retrieval["contexts"]`` (the list of individual chunk texts) is stored in
   the JSONL output and used by RAGAS for context_precision / context_recall.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from georgia_ev_intelligence.runtime_pipeline.generation.llm_client import generate_answer

from .interfaces import PromptBuilder


@dataclass(frozen=True)
class OnlyRagPromptBuilder:
    """Build the strict context-only prompt for the Only RAG pipeline."""

    def build(self, question: str, retrieved_context: str) -> str:
        return ONLY_RAG_PROMPT_TEMPLATE.format(
            retrieved_context=retrieved_context,
            user_question=question,
        )


class OnlyRagAnswerPipeline:
    """Generate an answer using only the provided retrieved context."""

    def __init__(
        self,
        prompt_builder: PromptBuilder | None = None,
        answer_generator: Callable[[str, int], str] = generate_answer,
    ) -> None:
        self._prompt_builder = prompt_builder or OnlyRagPromptBuilder()
        self._answer_generator = answer_generator

    def answer(
        self,
        question: str,
        retrieved_context: str,
        timeout: int = 180,
    ) -> str:
        prompt = self._prompt_builder.build(
            question=question,
            retrieved_context=retrieved_context,
        )
        return self._answer_generator(prompt, timeout)


ONLY_RAG_PROMPT_TEMPLATE = """You are answering questions about an EV supply chain knowledge base
for the state of Georgia.

Rules:
1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. Every factual claim must be supported by the context.
4. If the context does not contain the answer, say:
   "The provided knowledge base does not contain enough information to answer this."
5. Do not guess, infer from general knowledge, or fill missing details.

Provided context:
{retrieved_context}

User question:
{user_question}

Answer:"""
