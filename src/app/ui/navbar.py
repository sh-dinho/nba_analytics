from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Global Navigation Bar
# File: src/app/ui/navbar.py
# ============================================================
import streamlit as st


def _get_pages_safe():
    """
    Streamlit changed the pages API several times.
    This function safely retrieves the page list across versions.
    """
    # Newer versions (1.32+)
    if hasattr(st, "runtime") and hasattr(st.runtime, "get_pages"):
        return st.runtime.get_pages()

    # Older versions (1.20–1.31)
    if hasattr(st, "_runtime") and hasattr(st._runtime, "get_pages"):
        return st._runtime.get_pages()

    # Fallback: no pages detected
    return {}


def render_navbar():
    pages = _get_pages_safe()
    current = st.session_state.get("current_page", "")

    st.markdown(
        """
        <style>
            .nav-container {
                display: flex;
                gap: 18px;
                padding: 12px 0 18px 0;
                border-bottom: 1px solid #444;
                font-size: 16px;
            }
            .nav-item {
                text-decoration: none;
                color: #ccc;
                padding: 6px 14px;
                border-radius: 6px;
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

    # Convert dict → list of page objects
    page_list = list(pages.values())

    # Sort by script path (ensures 7_, 8_, 9_ order)
    page_list = sorted(page_list, key=lambda p: p["page_script_path"])

    for p in page_list:
        name = p["page_name"]
        url = "/" + p["url_path"]
        active = "nav-active" if current == name else ""
        nav_html += f'<a class="nav-item {active}" href="{url}">{name}</a>'

    nav_html += "</div>"

    st.markdown(nav_html, unsafe_allow_html=True)
