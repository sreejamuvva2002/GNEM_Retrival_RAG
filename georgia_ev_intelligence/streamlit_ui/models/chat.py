"""Chat domain models."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class Citation:
    """A clickable [n] citation that maps to a source id."""

    id: int
    source_id: str


@dataclass
class Message:
    """A single chat message (user or assistant)."""

    id: str
    role: str
    content: str
    timestamp: str
    citations: List[Citation] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)


@dataclass
class ChatHistoryEntry:
    """One conversation summary in the sidebar history list."""

    id: str
    title: str
    preview: str
    timestamp: str
    message_count: int


SUGGESTED_QUESTIONS: tuple = (
    "Which EV companies operate in Georgia?",
    "Show battery manufacturers in Georgia",
    "What suppliers support Hyundai's EV ecosystem?",
    "What charging infrastructure companies are active in Georgia?",
)


@dataclass
class Settings:
    """User-facing display settings."""

    show_citations: bool = True
    show_confidence: bool = True
    compact_mode: bool = False
    is_dark_mode: bool = True


def make_message_id() -> str:
    return f"msg-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
