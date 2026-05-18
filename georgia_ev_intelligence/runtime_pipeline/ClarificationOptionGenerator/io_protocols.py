"""I/O protocols for clarification prompting (terminal, web, API)."""
from __future__ import annotations

from typing import Protocol

from .models import ClarificationRequest, ClarificationSubmission


class ClarificationPrompterProtocol(Protocol):
    """Collect user clarification answers from any UI backend."""

    def prompt(self, request: ClarificationRequest) -> ClarificationSubmission:
        ...
