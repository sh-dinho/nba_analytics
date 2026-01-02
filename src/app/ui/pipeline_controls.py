from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import date, timedelta

from loguru import logger

from src.app.utils.pipeline_trigger import trigger_full_pipeline
from src.model.prediction.run_predictions import run_prediction_for_date
from src.config.paths import DATA_DIR


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _load_predictions_for_date(d: date) -> pd.DataFrame:
    """Load ML + totals + spread predictions for a given date."""
    base = DATA_DIR / "predictions"

    files = {
        "moneyline": base / f"moneyline_{d}.parquet",
        "totals": base / f"totals_{d}.parquet",
        "spread": base / f"spread_{d}.parquet",
    }

    dfs = []
    for label, path in files.items():
        if path.exists():
            df = pd.read_parquet(path)
            df["prediction_type"] = label
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# ------------------------------------------------------------
# Main UI Component
# ------------------------------------------------------------
def render_pipeline_controls() -> None:
    st.subheader("⚙️ Pipeline Controls")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # Predict Today Button
    # --------------------------------------------------------
    with col1:
        if st.button("🔮 Predict Today"):
            try:
                run_prediction_for_date(date.today())
                st.success("Predictions generated for today.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # --------------------------------------------------------
    # Run Full Pipeline Button
    # --------------------------------------------------------
    with col2:
        if st.button("🚀 Run Full Pipeline"):
            result = trigger_full_pipeline()
            if "successfully" in result.lower():
                st.success(result)
            else:
                st.error(result)

    st.divider()

    # --------------------------------------------------------
    # Backfill Panel
    # --------------------------------------------------------
    st.subheader("📅 Backfill Predictions")

    days = st.number_input(
        "Number of past days to backfill",
        min_value=1,
        max_value=60,
        value=7,
        step=1,
    )

    if st.button("Run Backfill"):
        today = date.today()
        for i in range(days):
            d = today - timedelta(days=i)
            try:
                run_prediction_for_date(d)
                st.success(f"Predictions generated for {d}")
            except Exception as e:
                st.error(f"Failed for {d}: {e}")

    st.divider()

    # --------------------------------------------------------
    # View Today's Predictions
    # --------------------------------------------------------
    st.subheader("📊 View Today’s Predictions")

    df = _load_predictions_for_date(date.today())

    if df.empty:
        st.warning("No predictions found for today.")
    else:
        st.dataframe(
            df.sort_values(["prediction_type", "game_id", "team"]),
            use_container_width=True,
        )