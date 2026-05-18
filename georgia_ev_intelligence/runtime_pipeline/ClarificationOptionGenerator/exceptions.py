"""Clarification module exceptions."""
from __future__ import annotations


class ClarificationError(Exception):
    """Base exception for clarification handling."""


class ClarificationNotFoundError(ClarificationError):
    """Raised when a clarification session id is not found."""


class ClarificationAlreadyResolvedError(ClarificationError):
    """Raised when resolving an already resolved session."""


class InvalidClarificationAnswerError(ClarificationError):
    """Raised when a submitted answer is invalid."""


class MissingCustomClarificationTextError(ClarificationError):
    """Raised when custom option is selected without custom_text."""


class ConceptRegistryError(ClarificationError):
    """Raised when concept registry data is invalid."""


class ClarificationCancelledError(ClarificationError):
    """Raised when the user cancels clarification input."""
