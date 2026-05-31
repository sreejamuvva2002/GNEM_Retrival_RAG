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
