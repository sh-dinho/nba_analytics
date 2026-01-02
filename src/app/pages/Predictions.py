from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Page: Predictions
# Purpose: View model predictions and edges for today / a date.
# ============================================================

from datetime import date
import pandas as pd
import streamlit as st

from src.app.ui.header import render_header
from src.app.ui.navbar import render_navbar
from src.app.ui.floating_action_bar import render_floating_action_bar
from src.app.ui.page_state import set_active_page
from src.config.paths import COMBINED_PRED_DIR


# ------------------------------------------------------------
# Load Predictions (Canonical v5.0)
# ------------------------------------------------------------
def _load_predictions(pred_date: date) -> pd.DataFrame:
    """
    Load canonical combined predictions for a given date.
    v5.0 uses COMBINED_PRED_DIR / predictions_YYYY-MM-DD.parquet
    """

    path = COMBINED_PRED_DIR / f"predictions_{pred_date}.parquet"

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        st.error(f"Failed to load predictions: {e}")
        return pd.DataFrame()

    # Ensure canonical columns exist
    required = {"game_id", "team", "opponent", "market", "model_prob", "implied_prob"}
    missing = required - set(df.columns)
    if missing:
        st.warning(f"Prediction file missing expected columns: {missing}")

    return df


# ------------------------------------------------------------
# Main Page
# ------------------------------------------------------------
def main() -> None:
    set_active_page("Predictions")

    render_header()
    render_navbar()

    st.title("🔮 Model Predictions")

    # --------------------------------------------------------
    # Date Selector + Reload
    # --------------------------------------------------------
    col_date, col_refresh = st.columns([3, 1])
    with col_date:
        target_date = st.date_input("Prediction Date", value=date.today())

    with col_refresh:
        if st.button("Reload Data"):
            st.experimental_rerun()

    # --------------------------------------------------------
    # Load Predictions
    # --------------------------------------------------------
    df = _load_predictions(target_date)

    if df.empty:
        st.warning(f"No prediction file found for {target_date}.")
        render_floating_action_bar()
        return

    st.success(f"Loaded {len(df):,} prediction rows for {target_date}.")

    # --------------------------------------------------------
    # Display Table
    # --------------------------------------------------------
    st.dataframe(df, use_container_width=True)

    # --------------------------------------------------------
    # Edge Distribution
    # --------------------------------------------------------
    if "edge" in df.columns:
        st.subheader("Edge Distribution")
        st.bar_chart(df["edge"])

    # --------------------------------------------------------
    # Floating Action Bar
    # --------------------------------------------------------
    render_floating_action_bar()


if __name__ == "__main__":
    main()
