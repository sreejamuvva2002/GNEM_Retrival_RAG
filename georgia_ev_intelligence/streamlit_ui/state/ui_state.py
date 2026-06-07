"""UI layout state (panels, view mode, selected items)."""
from __future__ import annotations

from . import session


VIEW_MODE_CHAT = "chat"
VIEW_MODE_SPLIT = "split"
VIEW_MODE_MAP = "map"

_VALID_VIEW_MODES = (VIEW_MODE_CHAT, VIEW_MODE_SPLIT, VIEW_MODE_MAP)


def initialize() -> None:
    session.ensure("sources_panel_open", False)
    session.ensure("settings_open", False)
    session.ensure("view_mode", VIEW_MODE_CHAT)
    session.ensure("selected_source_id", None)
    session.ensure("pending_query", None)
    session.ensure("pending_chat_memory", None)


def pending_query() -> str | None:
    """The query awaiting dispatch (user bubble already shown, answer pending)."""
    return session.get("pending_query")


def set_pending_query(value: str) -> None:
    session.set("pending_query", value)


def pending_chat_memory():
    return session.get("pending_chat_memory")


def set_pending_chat_memory(value) -> None:
    session.set("pending_chat_memory", value)


def pending_chat_history() -> list[dict[str, str]] | None:
    memory = pending_chat_memory()
    return getattr(memory, "recent_messages", None)


def set_pending_chat_history(value: list[dict[str, str]] | None) -> None:
    from ..models.chat import ChatMemory

    set_pending_chat_memory(ChatMemory(recent_messages=list(value or [])))


def clear_pending_query() -> None:
    session.set("pending_query", None)
    session.set("pending_chat_memory", None)


def sources_panel_open() -> bool:
    return bool(session.get("sources_panel_open", False))


def set_sources_panel_open(value: bool) -> None:
    session.set("sources_panel_open", bool(value))


def toggle_sources_panel() -> bool:
    return session.toggle("sources_panel_open", False)


def settings_open() -> bool:
    return bool(session.get("settings_open", False))


def set_settings_open(value: bool) -> None:
    session.set("settings_open", bool(value))


def view_mode() -> str:
    mode = session.get("view_mode", VIEW_MODE_CHAT)
    return mode if mode in _VALID_VIEW_MODES else VIEW_MODE_CHAT


def set_view_mode(mode: str) -> None:
    if mode in _VALID_VIEW_MODES:
        session.set("view_mode", mode)


def selected_source_id() -> str | None:
    return session.get("selected_source_id")


def set_selected_source_id(value: str | None) -> None:
    session.set("selected_source_id", value)
