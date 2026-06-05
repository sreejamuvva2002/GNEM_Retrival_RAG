"""Chat domain models."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List


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


@dataclass
class ChatMemory:
    """Conversation memory passed into RAG prompts for follow-up resolution."""

    summary: str = ""
    recent_messages: List[dict[str, str]] = field(default_factory=list)


@dataclass
class ChatTurnMetadata:
    """Session-only diagnostics for one completed RAG turn."""

    original_query: str
    effective_query: str
    history_used: bool
    source_ids: List[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


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
    # Light on first load — matches settings_state.initialize() so both defaults agree.
    is_dark_mode: bool = False


def make_message_id() -> str:
    return f"msg-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
