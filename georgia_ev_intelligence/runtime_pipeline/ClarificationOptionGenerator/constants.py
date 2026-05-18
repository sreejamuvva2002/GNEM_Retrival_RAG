"""Generic clarification constants (no KB-specific values)."""
from __future__ import annotations

from .models import ClarificationOption

DEFAULT_MAX_CLARIFICATION_DEPTH = 1

QUESTION_TEMPLATE_KNOWN = 'How should I interpret "{term}"?'
QUESTION_TEMPLATE_UNKNOWN = 'I found one unclear phrase: "{term}". How should I interpret it?'

DEFAULT_UNKNOWN_TERM_OPTIONS: tuple[ClarificationOption, ...] = (
    ClarificationOption(
        id="custom",
        label="Explain this phrase in your own words",
        meaning="The user will provide a custom definition.",
        requires_custom_text=True,
        ignore_term=False,
    ),
    ClarificationOption(
        id="ignore",
        label="Ignore this phrase",
        meaning="Do not use this ambiguous phrase in final interpretation.",
        requires_custom_text=False,
        ignore_term=True,
    ),
)

STATUS_PENDING = "pending"
STATUS_RESOLVED = "resolved"
STATUS_EXPIRED = "expired"
