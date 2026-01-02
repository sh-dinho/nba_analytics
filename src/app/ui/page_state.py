from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Page State
# File: src/app/ui/page_state.py
# Purpose:
#     Provide a stable, backwards‑compatible way to store and
#     retrieve the active page name in Streamlit session_state.
#
#     Notes:
#       • v5 primarily relies on Streamlit's internal page context
#       • This helper remains for legacy pages and manual overrides
# ============================================================

import streamlit as st


def set_active_page(name: str) -> None:
    """
    Set the active page name in session_state.

    This is used as a fallback when Streamlit's internal page
    context is unavailable (older versions, embedded views, or
    custom routing scenarios).
    """
    if not isinstance(name, str):
        raise TypeError("Page name must be a string.")

    st.session_state["current_page"] = name


def get_active_page() -> str:
    """
    Retrieve the active page name from session_state.

    Returns:
        str: The last known active page, or "Home" if unset.
    """
    return st.session_state.get("current_page", "Home")