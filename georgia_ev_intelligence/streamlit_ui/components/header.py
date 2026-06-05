"""Chat-panel header — GNEM logo badge (tooltip on hover) + title + subtitle.

Replica of the header inside chat-interface-with-map/chat-panel.tsx.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

_ASSETS = Path(__file__).resolve().parent.parent / "assets"


@lru_cache(maxsize=4)
def _data_uri(filename: str) -> str:
    """Return a base64 data URI for an asset (cached; Streamlit's HTML
    sanitizer reliably keeps inline <img> data URIs but can drop static paths)."""
    raw = (_ASSETS / filename).read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def render() -> None:
    # The tooltip text lives in a sibling <span> rather than a `data-*` attr —
    # Streamlit's HTML sanitizer drops custom data-attributes, which made the
    # earlier `content: attr(data-tooltip)` CSS resolve to an empty string.
    logo = _data_uri("gnem_logo.png")
    logo2x = _data_uri("gnem_logo@2x.png")
    st.markdown(
        f"""
        <div class="chat-header">
            <div class="gnem-logo-wrap" tabindex="0"
                 aria-label="Georgia Network for Electric Mobility">
                <img class="gnem-logo" src="{logo}" srcset="{logo} 1x, {logo2x} 2x"
                     alt="Georgia Network for Electric Mobility logo"
                     width="40" height="40" decoding="async" />
                <span class="gnem-tooltip">Georgia Network for Electric Mobility</span>
            </div>
            <div class="chat-header__titles">
                <div class="chat-header__title">Chat Assistant</div>
                <div class="chat-header__subtitle">Georgia EV Supply Chain Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
