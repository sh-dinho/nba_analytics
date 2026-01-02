from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Page: Home
# Purpose: Landing page for the analytics workstation.
# ============================================================

import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.navbar import render_navbar
from src.app.ui.floating_action_bar import render_floating_action_bar
from src.app.ui.page_state import set_active_page


def main() -> None:
    # --------------------------------------------------------
    # Page State
    # --------------------------------------------------------
    set_active_page("Home")

    # --------------------------------------------------------
    # Global UI Layout
    # --------------------------------------------------------
    render_header()
    render_navbar()

    st.divider()

    # --------------------------------------------------------
    # Home Content
    # --------------------------------------------------------
    st.title("🏀 NBA Analytics Workstation")
    st.subheader("Version 5 — Canonical Architecture")

    st.info(
        "Your analytics engine is running on a unified, version‑agnostic pipeline "
        "with stable ingestion, prediction, monitoring, and data quality validation."
    )

    st.markdown(
        """
        ### Quick Links
        - **Predictions** → Today’s model outputs and betting edges  
        - **Data Quality** → Canonical data health and schema checks  
        - **Backtest** → Historical performance and bet logs  
        - **Monitoring** → Pipeline and ingestion health  
        - **Parlay Builder** → Build and evaluate parlays  
        - **Bet Tracker** → Track wager history and outcomes  
        """
    )

    # --------------------------------------------------------
    # Floating Action Bar (global shortcuts)
    # --------------------------------------------------------
    render_floating_action_bar()

    # --------------------------------------------------------
    # Optional Footer
    # --------------------------------------------------------
    st.markdown(
        "<p style='text-align:center; color:gray; font-size:0.85rem;'>"
        "NBA Analytics Workstation v5.0 — Canonical Architecture"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
