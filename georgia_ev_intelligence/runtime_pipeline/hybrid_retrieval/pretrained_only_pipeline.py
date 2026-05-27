"""Answer generation that intentionally receives no retrieved context."""
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
