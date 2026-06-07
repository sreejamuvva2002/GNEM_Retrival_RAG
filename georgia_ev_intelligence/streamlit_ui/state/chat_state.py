"""Chat-related session state.

The production app uses one active in-session chat. It keeps the full visible
message list, a running summary for older turns, and the latest dispatch for
map/source rendering. The legacy history/store helpers remain for the capture
harness and sidebar component, but production does not depend on them.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models.chat import (
    ChatHistoryEntry,
    ChatMemory,
    ChatTurnMetadata,
    Message,
    make_message_id,
)
from . import session


_KEY_MESSAGES = "chat__messages"
_KEY_HISTORY = "chat__history"
_KEY_CURRENT_ID = "chat__current_id"
_KEY_LAST_DISPATCH = "chat__last_dispatch"
_KEY_CHAT_STORE = "chat__store"
_KEY_CONVERSATION_SUMMARY = "chat__conversation_summary"
_KEY_SUMMARY_CURSOR = "chat__summary_cursor"
_KEY_TURNS = "chat__turns"


def initialize() -> None:
    session.ensure(_KEY_MESSAGES, [])
    session.ensure(_KEY_HISTORY, [])
    session.ensure(_KEY_CURRENT_ID, None)
    session.ensure(_KEY_LAST_DISPATCH, None)
    session.ensure(_KEY_CHAT_STORE, {})
    session.ensure(_KEY_CONVERSATION_SUMMARY, "")
    session.ensure(_KEY_SUMMARY_CURSOR, 0)
    session.ensure(_KEY_TURNS, [])


def messages() -> List[Message]:
    return list(session.get(_KEY_MESSAGES, []))


def _clean_message_dicts(
    source_messages: list[Message],
    max_chars_per_message: int = 900,
) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for message in source_messages:
        role = getattr(message, "role", "")
        if role not in {"user", "assistant"}:
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        if max_chars_per_message > 0 and len(content) > max_chars_per_message:
            content = content[:max_chars_per_message].rstrip()
        cleaned.append({"role": role, "content": content})
    return cleaned


def rag_memory(recent_turns: int = 4, max_chars_per_message: int = 900) -> ChatMemory:
    """Conversation memory formatted for follow-up rewriting.

    Call this before appending the current user message so the latest question
    stays separate from the prior conversation. Older turns are represented by
    `conversation_summary`; the latest `recent_turns` are kept raw.
    """
    max_messages = max(0, int(recent_turns)) * 2
    if max_messages <= 0:
        return ChatMemory(summary=conversation_summary(), recent_messages=[])

    recent = _clean_message_dicts(messages(), max_chars_per_message=max_chars_per_message)
    return ChatMemory(
        summary=conversation_summary(),
        recent_messages=recent[-max_messages:],
    )


def rag_history(max_turns: int = 4, max_chars_per_message: int = 900) -> list[dict[str, str]]:
    """Backward-compatible recent-message history for older callers."""
    return rag_memory(
        recent_turns=max_turns,
        max_chars_per_message=max_chars_per_message,
    ).recent_messages


def append_message(message: Message) -> None:
    msgs = list(session.get(_KEY_MESSAGES, []))
    msgs.append(message)
    session.set(_KEY_MESSAGES, msgs)


def reset_messages() -> None:
    session.set(_KEY_MESSAGES, [])


def reset_conversation() -> None:
    session.set(_KEY_MESSAGES, [])
    session.set(_KEY_LAST_DISPATCH, None)
    session.set(_KEY_CONVERSATION_SUMMARY, "")
    session.set(_KEY_SUMMARY_CURSOR, 0)
    session.set(_KEY_TURNS, [])


def conversation_summary() -> str:
    return str(session.get(_KEY_CONVERSATION_SUMMARY, "") or "").strip()


def set_conversation_summary(value: str) -> None:
    session.set(_KEY_CONVERSATION_SUMMARY, str(value or "").strip())


def summary_cursor() -> int:
    try:
        return max(0, int(session.get(_KEY_SUMMARY_CURSOR, 0) or 0))
    except (TypeError, ValueError):
        return 0


def set_summary_cursor(value: int) -> None:
    session.set(_KEY_SUMMARY_CURSOR, max(0, int(value)))


def turns() -> List[ChatTurnMetadata]:
    return list(session.get(_KEY_TURNS, []))


def append_turn_metadata(turn: ChatTurnMetadata) -> None:
    existing = list(session.get(_KEY_TURNS, []))
    existing.append(turn)
    session.set(_KEY_TURNS, existing)


def summary_compaction_batch(
    recent_turns: int = 4,
    max_chars_per_message: int = 900,
) -> tuple[list[dict[str, str]], int]:
    """Return unsummarized completed messages older than the raw-memory window.

    The returned cursor is the message index to store only after summarization
    succeeds. The cursor tracks raw messages, not cleaned message dictionaries.
    """
    all_messages = messages()
    if not all_messages:
        return [], summary_cursor()

    raw_window = max(0, int(recent_turns)) * 2
    cutoff = max(0, len(all_messages) - raw_window)
    # Keep complete user/assistant pairs in the summarized region.
    if cutoff % 2:
        cutoff -= 1

    cursor = min(summary_cursor(), cutoff)
    if cursor % 2:
        cursor -= 1

    if cutoff <= cursor:
        return [], cursor

    batch = _clean_message_dicts(
        all_messages[cursor:cutoff],
        max_chars_per_message=max_chars_per_message,
    )
    return batch, cutoff


def history() -> List[ChatHistoryEntry]:
    return list(session.get(_KEY_HISTORY, []))


def _store_snapshot(chat_id: str) -> None:
    """Write the current messages + last_dispatch into the store under chat_id."""
    if not chat_id:
        return
    store = dict(session.get(_KEY_CHAT_STORE, {}) or {})
    store[chat_id] = {
        "messages": list(session.get(_KEY_MESSAGES, [])),
        "last_dispatch": session.get(_KEY_LAST_DISPATCH),
        "conversation_summary": session.get(_KEY_CONVERSATION_SUMMARY, ""),
        "summary_cursor": session.get(_KEY_SUMMARY_CURSOR, 0),
        "turns": list(session.get(_KEY_TURNS, [])),
    }
    session.set(_KEY_CHAT_STORE, store)


def _store_restore(chat_id: str) -> None:
    """Load messages + last_dispatch from the store for chat_id (or reset if absent)."""
    snapshot: Dict[str, Any] = (session.get(_KEY_CHAT_STORE, {}) or {}).get(chat_id) or {}
    session.set(_KEY_MESSAGES, list(snapshot.get("messages", [])))
    session.set(_KEY_LAST_DISPATCH, snapshot.get("last_dispatch"))
    session.set(_KEY_CONVERSATION_SUMMARY, snapshot.get("conversation_summary", ""))
    session.set(_KEY_SUMMARY_CURSOR, snapshot.get("summary_cursor", 0))
    session.set(_KEY_TURNS, list(snapshot.get("turns", [])))


def _store_drop(chat_id: str) -> None:
    store = dict(session.get(_KEY_CHAT_STORE, {}) or {})
    if chat_id in store:
        del store[chat_id]
        session.set(_KEY_CHAT_STORE, store)


def add_history_entry(title: str, preview: str, message_count: int) -> str:
    entry = ChatHistoryEntry(
        id=f"chat-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        title=title,
        preview=preview,
        timestamp=datetime.now().isoformat(),
        message_count=message_count,
    )
    entries = list(session.get(_KEY_HISTORY, []))
    entries.insert(0, entry)
    session.set(_KEY_HISTORY, entries)
    session.set(_KEY_CURRENT_ID, entry.id)
    _store_snapshot(entry.id)
    return entry.id


def remove_history_entry(entry_id: str) -> None:
    entries = [e for e in session.get(_KEY_HISTORY, []) if e.id != entry_id]
    session.set(_KEY_HISTORY, entries)
    _store_drop(entry_id)
    if session.get(_KEY_CURRENT_ID) == entry_id:
        session.set(_KEY_CURRENT_ID, None)
        reset_conversation()


def current_chat_id() -> Optional[str]:
    return session.get(_KEY_CURRENT_ID)


def set_current_chat_id(value: Optional[str]) -> None:
    outgoing = session.get(_KEY_CURRENT_ID)
    if outgoing and outgoing != value:
        _store_snapshot(outgoing)
    session.set(_KEY_CURRENT_ID, value)
    if value:
        _store_restore(value)


def last_dispatch():
    return session.get(_KEY_LAST_DISPATCH)


def set_last_dispatch(value) -> None:
    session.set(_KEY_LAST_DISPATCH, value)


def start_new_chat() -> None:
    from . import ui_state

    outgoing = session.get(_KEY_CURRENT_ID)
    if outgoing:
        _store_snapshot(outgoing)
    reset_conversation()
    session.set(_KEY_CURRENT_ID, None)
    ui_state.set_sources_panel_open(False)


def make_user_message(content: str) -> Message:
    return Message(
        id=make_message_id(),
        role="user",
        content=content,
        timestamp=datetime.now().isoformat(),
    )


def make_assistant_message(content: str, source_ids: List[str]) -> Message:
    return Message(
        id=make_message_id(),
        role="assistant",
        content=content,
        timestamp=datetime.now().isoformat(),
        source_ids=list(source_ids),
    )
