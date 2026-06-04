"""Chat-related session state.

Persists per-chat snapshots in `_KEY_CHAT_STORE` so clicking "Open" on a prior
history entry restores its messages + dispatch instead of just changing the
active id with no visible effect.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models.chat import ChatHistoryEntry, Message, make_message_id
from . import session


_KEY_MESSAGES = "chat__messages"
_KEY_HISTORY = "chat__history"
_KEY_CURRENT_ID = "chat__current_id"
_KEY_LAST_DISPATCH = "chat__last_dispatch"
_KEY_CHAT_STORE = "chat__store"


def initialize() -> None:
    session.ensure(_KEY_MESSAGES, [])
    session.ensure(_KEY_HISTORY, [])
    session.ensure(_KEY_CURRENT_ID, None)
    session.ensure(_KEY_LAST_DISPATCH, None)
    session.ensure(_KEY_CHAT_STORE, {})


def messages() -> List[Message]:
    return list(session.get(_KEY_MESSAGES, []))


def append_message(message: Message) -> None:
    msgs = list(session.get(_KEY_MESSAGES, []))
    msgs.append(message)
    session.set(_KEY_MESSAGES, msgs)


def reset_messages() -> None:
    session.set(_KEY_MESSAGES, [])


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
    }
    session.set(_KEY_CHAT_STORE, store)


def _store_restore(chat_id: str) -> None:
    """Load messages + last_dispatch from the store for chat_id (or reset if absent)."""
    snapshot: Dict[str, Any] = (session.get(_KEY_CHAT_STORE, {}) or {}).get(chat_id) or {}
    session.set(_KEY_MESSAGES, list(snapshot.get("messages", [])))
    session.set(_KEY_LAST_DISPATCH, snapshot.get("last_dispatch"))


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
        reset_messages()
        session.set(_KEY_LAST_DISPATCH, None)


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
    reset_messages()
    session.set(_KEY_CURRENT_ID, None)
    session.set(_KEY_LAST_DISPATCH, None)
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
