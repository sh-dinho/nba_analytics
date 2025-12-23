# ============================================================
# 🏀 NBA Analytics v4
# Module: Model Monitoring Dashboard
# File: src/dashboard/model_monitor.py
# Author: Sadiq
#
# Description:
#     Monitors model health, drift, performance, and lineage.
#     Includes:
#       - registry overview
#       - lineage per model type
#       - drift detection (rolling mean)
#       - staleness alerts
# ============================================================

from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt

from src.model.registry import _load_index
from src.config.paths import COMBINED_PRED_DIR


def model_monitor_view():
    st.title("🧠 Model Monitoring Dashboard (v4)")

    # ------------------------------------------------------------
    # Load model registry
    # ------------------------------------------------------------
    index = _load_index()
    models = index.get("models", [])

    if not models:
        st.warning("No models found in registry.")
        return

    df = pd.DataFrame(models)
    df = df.sort_values("created_at_utc", ascending=False)

    st.subheader("📦 Model Registry Overview")
    st.dataframe(df, use_container_width=True)

    # ------------------------------------------------------------
    # Select model type
    # ------------------------------------------------------------
    model_type = st.selectbox("Select model type", df["model_type"].unique())
    mdf = df[df["model_type"] == model_type]

    st.subheader(f"🧬 {model_type.capitalize()} Model Lineage")
    st.dataframe(mdf, use_container_width=True)

    # ------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------
    st.subheader("📉 Feature Drift (Rolling Mean)")

    try:
        pred_files = sorted(COMBINED_PRED_DIR.glob("combined_*.parquet"))[-30:]
        if not pred_files:
            st.info("No combined prediction files found.")
            return

        preds = pd.concat([pd.read_parquet(f) for f in pred_files], ignore_index=True)

        # Ensure date is datetime
        if "date" in preds.columns:
            preds["date"] = pd.to_datetime(preds["date"])
        else:
            st.warning("Combined predictions missing 'date' column.")
            return

        # Identify drift-sensitive columns
        drift_cols = [
            c
            for c in preds.columns
            if any(k in c for k in ["win_probability", "predicted", "elo"])
        ]

        if not drift_cols:
            st.info("No drift-sensitive columns found.")
            return

        # Rolling mean (7-day window)
        drift_df = (
            preds.groupby("date")[drift_cols]
            .mean()
            .rolling(window=7, min_periods=1)
            .mean()
            .reset_index()
        )

        # Build multi-row chart
        chart = (
            alt.Chart(drift_df)
            .mark_line()
            .encode(
                x="date:T",
                y=alt.Y(alt.repeat("row"), type="quantitative"),
                color=alt.value("#1f77b4"),
            )
            .properties(height=120)
            .repeat(row=drift_cols)
            .resolve_scale(y="independent")
        )

        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.warning(f"Drift charts unavailable: {e}")

    # ------------------------------------------------------------
    # Model staleness
    # ------------------------------------------------------------
    st.subheader("⏳ Model Freshness")

    latest = mdf.iloc[0]
    train_end = latest.get("train_end_date")

    if train_end:
        st.metric("Last Training Date", train_end)
    else:
        st.info("No training date recorded.")

    st.caption("A model is considered stale if not retrained for > 14 days.")
