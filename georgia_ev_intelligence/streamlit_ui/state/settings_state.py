"""User-display settings state."""
from __future__ import annotations

from ..models.chat import Settings
from . import session


def initialize() -> None:
    session.ensure("settings__show_citations", True)
    session.ensure("settings__show_confidence", True)
    session.ensure("settings__compact_mode", False)
    session.ensure("settings__is_dark_mode", True)


def settings() -> Settings:
    return Settings(
        show_citations=bool(session.get("settings__show_citations", True)),
        show_confidence=bool(session.get("settings__show_confidence", True)),
        compact_mode=bool(session.get("settings__compact_mode", False)),
        is_dark_mode=bool(session.get("settings__is_dark_mode", True)),
    )


def set_show_citations(value: bool) -> None:
    session.set("settings__show_citations", bool(value))


def set_show_confidence(value: bool) -> None:
    session.set("settings__show_confidence", bool(value))


def set_compact_mode(value: bool) -> None:
    session.set("settings__compact_mode", bool(value))


def set_dark_mode(value: bool) -> None:
    session.set("settings__is_dark_mode", bool(value))


def toggle_theme() -> None:
    session.toggle("settings__is_dark_mode", True)
