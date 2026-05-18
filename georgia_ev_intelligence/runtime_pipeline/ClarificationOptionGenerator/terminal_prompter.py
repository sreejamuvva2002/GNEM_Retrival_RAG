"""Terminal UI for collecting clarification answers."""
from __future__ import annotations

from collections.abc import Callable

from .exceptions import ClarificationCancelledError
from .models import (
    AmbiguousTermClarification,
    ClarificationAnswer,
    ClarificationRequest,
    ClarificationSubmission,
)

_CANCEL_TOKENS = frozenset({"q", "quit", "exit", "cancel"})
_HEADER_WIDTH = 60


class TerminalClarificationPrompter:
    """Display clarification options and collect answers via terminal I/O."""

    def __init__(
        self,
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
    ) -> None:
        self._input = input_func
        self._output = output_func

    def prompt(self, request: ClarificationRequest) -> ClarificationSubmission:
        self._display_header(request)
        self._display_term_count(request)

        answers: list[ClarificationAnswer] = []
        for idx, term in enumerate(request.ambiguous_terms, start=1):
            self._output("")
            self._output(f'[{idx}] "{term.term}"')
            self._output("")
            self._output(term.question)
            self._output("")
            self._display_options(term)
            answer = self._prompt_for_term(term)
            answers.append(answer)

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

    def _display_term_count(self, request: ClarificationRequest) -> None:
        count = len(request.ambiguous_terms)
        label = "phrase" if count == 1 else "phrases"
        self._output(f"I found {count} unclear {label}:")
        self._output("")

    def _display_options(self, term: AmbiguousTermClarification) -> None:
        for idx, option in enumerate(term.options, start=1):
            self._output(f"  {idx}. {option.label}")
            self._output(f"     Meaning: {option.meaning}")
            self._output("")

    def _prompt_for_term(
        self, term: AmbiguousTermClarification
    ) -> ClarificationAnswer:
        option_id, custom_text = self._read_option(term)
        return ClarificationAnswer(
            term=term.term,
            selected_option_id=option_id,
            custom_text=custom_text,
        )

    def _read_option(
        self, term: AmbiguousTermClarification
    ) -> tuple[str, str | None]:
        prompt = f'Enter option number for "{term.term}": '
        while True:
            self._output(prompt)
            raw = self._input(prompt).strip()
            if raw.lower() in _CANCEL_TOKENS:
                raise ClarificationCancelledError(
                    "Clarification cancelled by user."
                )
            option = self._option_number_to_option(term, raw)
            if option is None:
                self._output(
                    f'Invalid option "{raw}". Enter a number between '
                    f"1 and {len(term.options)}, or q to quit."
                )
                continue
            if option.requires_custom_text or option.id == "custom":
                custom_text = self._read_custom_text(term.term)
                return option.id, custom_text
            return option.id, None

    def _read_custom_text(self, term: str) -> str:
        self._output("")
        while True:
            raw = self._input(f'Enter your definition for "{term}":\n> ').strip()
            if raw.lower() in _CANCEL_TOKENS:
                raise ClarificationCancelledError(
                    "Clarification cancelled by user."
                )
            if raw:
                return raw
            self._output("Definition cannot be empty. Please try again.")

    def _option_number_to_option(
        self,
        term: AmbiguousTermClarification,
        raw: str,
    ):
        try:
            number = int(raw)
        except ValueError:
            return None
        if number < 1 or number > len(term.options):
            return None
        return term.options[number - 1]
