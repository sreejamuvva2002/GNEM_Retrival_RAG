"""Top header bar — GNEM logo + title + view-mode toggle + theme/sources/settings."""
from __future__ import annotations

import streamlit as st

from ..state import settings_state, ui_state
from . import settings_panel


def render() -> None:
    st.markdown(
        """
        <div class="gnem-header">
            <div class="gnem-header__brand">
                <span class="gnem-logo">GNEM</span>
                <span class="gnem-header__title">Georgia EV Supply Chain Intelligence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bar_center, bar_right = st.columns([0.55, 0.45])
    with bar_center:
        view_modes = [
            ("chat", ":material/chat: Chat"),
            ("split", ":material/space_dashboard: Split"),
            ("map", ":material/map: Map"),
        ]
        current = ui_state.view_mode()
        labels = [label for _, label in view_modes]
        keys = [key for key, _ in view_modes]
        index = keys.index(current) if current in keys else 0
        chosen = st.segmented_control(
            "View mode",
            options=labels,
            default=labels[index],
            key="hdr_view_mode",
            label_visibility="collapsed",
        )
        if chosen is None:
            chosen = labels[index]
        ui_state.set_view_mode(keys[labels.index(chosen)])
    with bar_right:
        _spacer, theme_col, settings_col = st.columns([0.62, 0.23, 0.15])
        with theme_col:
            # Sliding sun/moon switch (sun/moon icons drawn on the track via CSS
            # scoped to .st-key-hdr_theme_toggle). ON = dark mode.
            is_dark = settings_state.settings().is_dark_mode
            toggled = st.toggle(
                "Theme",
                value=is_dark,
                key="hdr_theme_toggle",
                label_visibility="collapsed",
                help="Toggle dark mode" if not is_dark else "Toggle light mode",
            )
            if toggled != is_dark:
                settings_state.set_dark_mode(toggled)
                st.rerun()
        with settings_col:
            # Compact icon-only button (sized via .st-key-hdr_settings CSS).
            if st.button(":material/settings:", key="hdr_settings", help="Open settings"):
                settings_panel.open_settings()
