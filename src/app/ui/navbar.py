from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Name: Global Navigation Bar
# File: src/app/ui/navbar.py
# Purpose:
#     Render a stable, version‑agnostic horizontal navigation bar
#     with custom ordering, active highlighting, and responsive layout.
# ============================================================

import streamlit as st


# ------------------------------------------------------------
# Page Retrieval (Streamlit‑safe)
# ------------------------------------------------------------
def _get_pages_safe() -> dict:
    """
    Retrieve Streamlit pages across versions.
    Streamlit has changed this API multiple times, so we check all known paths.
    """
    candidates = [
        getattr(st, "runtime", None),
        getattr(st, "_runtime", None),
    ]

    for c in candidates:
        if c and hasattr(c, "get_pages"):
            try:
                return c.get_pages()
            except Exception:
                pass

    return {}


# ------------------------------------------------------------
# Active Page Detection
# ------------------------------------------------------------
def _detect_active_page(pages: dict) -> str:
    """
    Detect the active page using Streamlit's internal context.
    Fallback to session state if unavailable.
    """
    try:
        ctx = st.context
        script = ctx.page.page_script_path

        for p in pages.values():
            if p.get("page_script_path") == script:
                return p["page_name"]

    except Exception:
        pass

    # Fallback
    return st.session_state.get("current_page", "")


# ------------------------------------------------------------
# Canonical Ordering
# ------------------------------------------------------------
CANONICAL_ORDER = [
    "Home",
    "Predictions",
    "Data Quality",
    "Backtest",
    "Monitoring",
    "Parlay Builder",
]


def _sort_pages(pages: dict) -> list:
    """
    Sort pages using canonical order first, then alphabetical fallback.
    """
    def sort_key(p):
        name = p["page_name"]
        if name in CANONICAL_ORDER:
            return (0, CANONICAL_ORDER.index(name))
        return (1, name.lower())

    return sorted(list(pages.values()), key=sort_key)


# ------------------------------------------------------------
# Render Navbar
# ------------------------------------------------------------
def render_navbar() -> None:
    pages = _get_pages_safe()
    active = _detect_active_page(pages)
    page_list = _sort_pages(pages)

    # --------------------------------------------------------
    # Styles
    # --------------------------------------------------------
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

            @media (max-width: 600px) {
                .nav-container {
                    font-size: 14px;
                    gap: 10px;
                }
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

    # --------------------------------------------------------
    # HTML Build
    # --------------------------------------------------------
    nav_html = '<div class="nav-container">'

    for p in page_list:
        name = p["page_name"]
        url = "/" + p["url_path"]
        is_active = "nav-active" if name == active else ""
        nav_html += f'<a class="nav-item {is_active}" href="{url}">{name}</a>'

    nav_html += "</div>"

    st.markdown(nav_html, unsafe_allow_html=True)