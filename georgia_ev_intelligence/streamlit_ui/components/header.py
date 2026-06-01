"""Top header bar — GNEM logo + title + view-mode toggle + theme/sources/settings."""
from __future__ import annotations

import streamlit as st

from ..state import settings_state, ui_state


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
            ("chat", "💬 Chat"),
            ("split", "▥ Split"),
            ("map", "🗺 Map"),
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
        _spacer, theme_col, settings_col = st.columns([0.6, 0.25, 0.15])
        with theme_col:
            # Icon-only sun/moon switch — no text label.
            is_dark = settings_state.settings().is_dark_mode
            current_icon = "🌙" if is_dark else "☀"
            picked = st.segmented_control(
                "Theme",
                options=["☀", "🌙"],
                default=current_icon,
                key="hdr_theme_switch",
                label_visibility="collapsed",
            )
            if picked is not None and (picked == "🌙") != is_dark:
                settings_state.set_dark_mode(picked == "🌙")
                st.rerun()
        with settings_col:
            if st.button("⚙", key="hdr_settings", use_container_width=True, help="Settings"):
                ui_state.set_settings_open(True)
                st.rerun()
