# ============================================================
# 🏀 NBA Analytics v4
# Module: UI Top Navigation
# Author: Sadiq
# Version: 4.0.0
# Purpose: Handles horizontal tab-based navigation across pages.
# ============================================================
from __future__ import annotations
import streamlit as st


def _get_pages_safe():
    if hasattr(st, "runtime") and hasattr(st.runtime, "get_pages"):
        return st.runtime.get_pages()
    return {}


def render_top_tabs():
    pages = _get_pages_safe()
    current = st.session_state.get("current_page", "")
    page_list = sorted(list(pages.values()), key=lambda p: p["page_script_path"])

    tab_labels = [p["page_name"] for p in page_list]

    try:
        active_index = tab_labels.index(current)
    except ValueError:
        active_index = 0

    tabs = st.tabs(tab_labels)

    # FIXED: Replaced HTML meta-refresh with st.switch_page
    for i, tab in enumerate(tabs):
        with tab:
            if i != active_index:
                target_page = page_list[i]["page_script_path"]
                st.session_state["current_page"] = tab_labels[i]
                st.switch_page(target_page)
