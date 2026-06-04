"""Tiny typed wrapper around st.session_state.

We do not subclass SessionStateProxy — Streamlit doesn't make that easy. Instead
this is a thin helper that ensures defaults exist and exposes getters/setters
with clear types.
"""
from __future__ import annotations

from typing import Any

import streamlit as st


def ensure(key: str, default: Any) -> Any:
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set(key: str, value: Any) -> None:  # noqa: A001 — name mirrors the dict API
    st.session_state[key] = value


def toggle(key: str, default: bool = False) -> bool:
    new_value = not bool(st.session_state.get(key, default))
    st.session_state[key] = new_value
    return new_value
