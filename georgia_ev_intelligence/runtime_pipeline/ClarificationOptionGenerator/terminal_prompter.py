"""Terminal UI for collecting open-ended clarification answers."""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .exceptions import ClarificationCancelledError
from .models import (
    ClarificationAnswer,
    ClarificationRequest,
    ClarificationSubmission,
)

_CANCEL_TOKENS = frozenset({"q", "quit", "exit", "cancel"})
_HEADER_WIDTH = 60


class ClarificationPrompterProtocol(Protocol):
    """Collect user clarification answers from any UI backend."""

    def prompt(self, request: ClarificationRequest) -> ClarificationSubmission:
        ...


class TerminalClarificationPrompter:
    """Display open-ended clarification questions and collect free-text answers."""

    def __init__(
        self,
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
    ) -> None:
        self._input = input_func
        self._output = output_func

    def prompt(self, request: ClarificationRequest) -> ClarificationSubmission:
        self._display_header(request)
        self._display_question_count(request)

        answers: list[ClarificationAnswer] = []
        for idx, question in enumerate(request.questions, start=1):
            self._output("")
            self._output(f'[{idx}] "{question.term}"')
            self._output("")
            self._output(question.question)
            self._output("")
            text = self._read_clarification(question.term)
            answers.append(ClarificationAnswer(term=question.term, custom_text=text))

        return ClarificationSubmission(
            clarification_id=request.clarification_id,
            answers=answers,
        )

    def _display_header(self, request: ClarificationRequest) -> None:
        line = "=" * _HEADER_WIDTH
        self._output(line)
        self._output("Clarification Required")
        self._output(line)
        self._output("")
        self._output("Original query:")
        self._output(request.original_query)
        self._output("")

    def _display_question_count(self, request: ClarificationRequest) -> None:
        count = len(request.questions)
        label = "phrase" if count == 1 else "phrases"
        self._output(f"I found {count} unclear {label}:")

    def _read_clarification(self, term: str) -> str:
        prompt_text = "Enter your clarification:\n> "
        while True:
            raw = self._input(prompt_text).strip()
            if raw.lower() in _CANCEL_TOKENS:
                raise ClarificationCancelledError(
                    "Clarification cancelled by user."
                )
            if raw:
                return raw
            self._output("Clarification cannot be empty. Please try again.")
