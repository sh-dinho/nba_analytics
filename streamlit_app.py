from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics — Canonical Frontend Shell
# ============================================================

import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.navbar import render_navbar
from src.app.ui.page_state import get_active_page


# ------------------------------------------------------------
# Session Initialization
# ------------------------------------------------------------
def _init_session_state():
    defaults = {
        "parlay_legs": [],
        "current_page": "Home",
        "selected_date": None,
        "selected_team": None,
        "selected_model": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ------------------------------------------------------------
# Page Registry (canonical)
# ------------------------------------------------------------
def _render_home():
    st.title("🏀 NBA Analytics Dashboard")
    st.subheader("Version 5 — Canonical Architecture")

    st.info(
        "Your analytics engine is now running on a unified, version‑agnostic "
        "canonical pipeline with stable ingestion, feature generation, and "
        "thread‑safe data persistence."
    )

    st.success("Use the navigation bar to explore predictions, backtests, and dashboards.")


PAGE_REGISTRY = {
    "Home": _render_home,
    "Data Quality": lambda: __import__("src.app.pages.data_quality", fromlist=["render_data_quality_page"]).render_data_quality_page(),
    "Predictions": lambda: __import__("src.app.pages.predictions", fromlist=["render_predictions_page"]).render_predictions_page(),
    "Backtest": lambda: __import__("src.app.pages.backtest", fromlist=["render_backtest_page"]).render_backtest_page(),
    "Monitoring": lambda: __import__("src.app.pages.monitoring", fromlist=["render_monitoring_page"]).render_monitoring_page(),
    "Parlay Builder": lambda: __import__("src.app.pages.parlay_builder", fromlist=["render_parlay_builder_page"]).render_parlay_builder_page(),
}


# ------------------------------------------------------------
# Page Router
# ------------------------------------------------------------
def _render_page():
    page = get_active_page()
    renderer = PAGE_REGISTRY.get(page)

    if not renderer:
        st.error(f"Unknown page: {page}")
        return

    try:
        renderer()
    except Exception as e:
        st.error(f"An error occurred while rendering '{page}': {e}")


# ------------------------------------------------------------
# Main App Entry
# ------------------------------------------------------------
def main():
    _init_session_state()

    render_header()
    render_navbar()

    _render_page()


if __name__ == "__main__":
    main()
