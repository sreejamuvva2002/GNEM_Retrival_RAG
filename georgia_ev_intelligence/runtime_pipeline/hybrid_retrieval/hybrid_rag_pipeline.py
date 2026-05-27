"""Answer generation that combines retrieved context with pretrained knowledge.

WHY THIS FILE EXISTS
--------------------
Implements the ``hybrid_rag`` pipeline, which uses the SAME retrieved parent
chunks as ``rag_only`` but with a different prompt that allows the LLM to
also use its pretrained world knowledge to fill gaps.  This tests whether
allowing pretrained knowledge supplementation improves answer quality over
strict RAG.

DISTINCTION FROM rag_only
--------------------------
``rag_only``   → model MUST use ONLY retrieved context; refuses if context
                 is insufficient.
``hybrid_rag`` → model uses retrieved context as PRIMARY source but MAY
                 supplement with pretrained knowledge, explicitly labelling
                 any addition as ``[From general knowledge: ...]``.

PROMPT DESIGN (correctness-critical)
--------------------------------------
The prompt enforces:
  1. Retrieved context is PRIMARY — ground every claim there first.
  2. Pretrained supplement is LABELLED — any non-context fact must be
     wrapped in ``[From general knowledge: ...]``.
  3. Do not invent companies, roles, products, OEMs, etc. from either source.
  4. Style rules: one company per line, exact field formatting.

CONTEXT PASSED
--------------
Unlike ``pretrained_only``, this pipeline DOES receive retrieved context —
the same ``retrieval["formatted_context"]`` as ``rag_only``.
Verified in ``run_baseline.py::_answer_hybrid_rag``.

CORRECTNESS CONTRACT
--------------------
✅ Receives the same retrieved context list as rag_only (from RetrievalCache).
✅ ``contexts`` list stored in JSONL output is populated (used for RAGAS
   faithfulness scoring — whether answers are grounded in the retrieved text).
✅ DIFFERENT prompt from rag_only: the model is explicitly permitted and
   expected to use pretrained knowledge to supplement, unlike rag_only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from georgia_ev_intelligence.runtime_pipeline.generation.llm_client import generate_answer

from .interfaces import PromptBuilder


@dataclass(frozen=True)
class HybridRagPromptBuilder:
    """Build the context-primary + pretrained-supplement prompt."""

    def build(self, question: str, retrieved_context: str) -> str:
        return HYBRID_RAG_PROMPT_TEMPLATE.format(
            retrieved_context=retrieved_context,
            user_question=question,
        )


class HybridRagAnswerPipeline:
    """Generate an answer using retrieved context as primary source, supplemented
    by pretrained knowledge where the context is incomplete."""

    def __init__(
        self,
        prompt_builder: PromptBuilder | None = None,
        answer_generator: Callable[[str, int], str] = generate_answer,
    ) -> None:
        self._prompt_builder = prompt_builder or HybridRagPromptBuilder()
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


HYBRID_RAG_PROMPT_TEMPLATE = """You are an analyst answering questions about the Georgia EV supply chain.

You have access to two sources of knowledge:
1. **Retrieved context** — records pulled from the Georgia EV Supply Chain knowledge base (provided below).
2. **Your pretrained knowledge** — general knowledge you acquired during training.

Rules:
1. Use the retrieved context as your PRIMARY source. Ground every claim in it where possible.
2. You MAY supplement with pretrained knowledge ONLY where the context is clearly incomplete or
   does not address part of the question. Any information from pretrained knowledge must be
   explicitly labelled as:
   [From general knowledge: <the information>]
3. Apply every filter in the question exactly (tier, role, location, employment figures, OEM).
4. Each physical company is one entity — do not double-count a company that appears in multiple
   context records.
5. If a value is missing from the context for an item you would otherwise list, write "n/a" for
   that field rather than guessing.
6. Do not invent companies, roles, products, OEMs, locations, or employment numbers that are
   neither in the context nor genuinely supported by pretrained knowledge.
7. Format your answer following these style rules:
   - OPENING LINE: Begin with a one-sentence count or direct answer as appropriate.
   - BODY: One item per line. No bullets, no numbering. Format each line as:
       <Company Name> [<Tier>] | <FieldLabel>: <value> | <FieldLabel>: <value>
   - TONE: Direct, factual, concise. No preamble, no closing summary.

Retrieved context:
{retrieved_context}

User question:
{user_question}

Answer:"""
