from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Module: UI Top Navigation (Canonical)
# Purpose:
#     Provide tab-like navigation using buttons instead of
#     Streamlit tabs, avoiding rerun loops and instability.
# ============================================================

import streamlit as st
from src.app.ui.page_state import set_active_page, get_active_page


def render_top_tabs(page_names: list[str]) -> None:
    """
    Render a horizontal tab-like navigation bar using buttons.
    This avoids Streamlit rerun issues and works across versions.
    """

    active = get_active_page()

    cols = st.columns(len(page_names))

    for i, name in enumerate(page_names):
        is_active = (name == active)

        button_style = (
            "background-color: #2ecc71; color: black; font-weight: bold;"
            if is_active
            else "background-color: #333; color: #ccc;"
        )

        with cols[i]:
            if st.button(
                name,
                key=f"tab_{name}",
                help=f"Go to {name}",
                use_container_width=True,
            ):
                set_active_page(name)
                st.switch_page(f"{name}.py")  # canonical routing