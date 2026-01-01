from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Page State
# File: src/app/ui/page_state.py
# Purpose: Legacy helper for setting active page name in
#          session_state (navbar uses this as a fallback).
# ============================================================

import streamlit as st


def set_active_page(name: str) -> None:
    """
    Backwards-compatible helper. In v5, the navbar primarily uses
    Streamlit's internal page context, but this remains for legacy
    pages and manual overrides.
    """
    st.session_state["current_page"] = name
