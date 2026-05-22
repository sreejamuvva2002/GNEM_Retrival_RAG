"""Answer generation that is constrained to retrieved context only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from georgia_ev_intelligence.runtime_pipeline.generation.llm_client import generate_answer


class PromptBuilder(Protocol):
    """Build a prompt from a question and retrieved context."""

    def build(self, question: str, retrieved_context: str) -> str:
        """Return the prompt sent to the LLM."""


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
