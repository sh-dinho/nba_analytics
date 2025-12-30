from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Page State
# File: src/app/ui/page_state.py
# ============================================================

import streamlit as st


def set_active_page(name: str):
    st.session_state["current_page"] = name
