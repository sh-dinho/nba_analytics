from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Global Navigation Bar
# File: src/app/ui/navbar.py
# Purpose: Render horizontal navigation across Streamlit pages.
# ============================================================

import streamlit as st


def _get_pages_safe():
    """
    Safely retrieve Streamlit pages across versions.
    """
    if hasattr(st, "runtime") and hasattr(st.runtime, "get_pages"):
        return st.runtime.get_pages()
    if hasattr(st, "_runtime") and hasattr(st._runtime, "get_pages"):
        return st._runtime.get_pages()
    return {}


def _detect_active_page(pages):
    """
    Prefer Streamlit's internal context; fallback to session state.
    """
    try:
        ctx = st.context
        script = ctx.page.page_script_path
        for p in pages.values():
            if p["page_script_path"] == script:
                return p["page_name"]
    except Exception:
        pass

    return st.session_state.get("current_page", "")


def render_navbar() -> None:
    pages = _get_pages_safe()
    active = _detect_active_page(pages)

    page_list = sorted(list(pages.values()), key=lambda p: p["page_name"].lower())

    st.markdown(
        """
        <style>
            .nav-container {
                display: flex;
                flex-wrap: wrap;
                gap: 14px;
                padding: 12px 0 18px 0;
                border-bottom: 1px solid #444;
                font-size: 16px;
            }
            .nav-item {
                text-decoration: none;
                color: #ccc;
                padding: 6px 14px;
                border-radius: 6px;
                transition: 0.15s ease;
            }
            .nav-item:hover {
                background-color: #333;
                color: white;
            }
            .nav-active {
                background-color: #2ecc71;
                color: black !important;
                font-weight: bold;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    nav_html = '<div class="nav-container">'

    for p in page_list:
        name = p["page_name"]
        url = "/" + p["url_path"]
        is_active = "nav-active" if name == active else ""
        nav_html += f'<a class="nav-item {is_active}" href="{url}">{name}</a>'

    nav_html += "</div>"

    st.markdown(nav_html, unsafe_allow_html=True)
