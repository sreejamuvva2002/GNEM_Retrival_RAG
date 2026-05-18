"""Exceptions for the phrase classifier module."""
from __future__ import annotations


class PhraseClassifierError(Exception):
    """Base exception for phrase classification."""


class PhraseClassifierTimeoutError(PhraseClassifierError):
    """Raised when the LLM call times out."""


class PhraseClassifierParseError(PhraseClassifierError):
    """Raised when the LLM response cannot be parsed."""
