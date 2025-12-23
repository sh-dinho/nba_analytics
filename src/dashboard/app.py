from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st
from loguru import logger

from src.config.paths import (
    MONEYLINE_PRED_DIR,
    TOTALS_PRED_DIR,
    SPREAD_PRED_DIR,
    COMBINED_PRED_DIR,
)


# ------------------------------------------------------------
# Loaders
# ------------------------------------------------------------


def _load_moneyline(pred_date: date) -> pd.DataFrame:
    """
    Load moneyline predictions for a given date.

    Expected path pattern (v4):
      data/predictions_moneyline/moneyline_{date}.parquet
    """
    path = MONEYLINE_PRED_DIR / f"moneyline_{pred_date}.parquet"
    if not path.exists():
        logger.warning(f"[Dashboard] Moneyline file not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_totals(pred_date: date) -> pd.DataFrame:
    """
    Load totals predictions for a given date.

    Expected path pattern:
      data/predictions_totals/totals_{date}.parquet
    """
    path = TOTALS_PRED_DIR / f"totals_{pred_date}.parquet"
    if not path.exists():
        logger.warning(f"[Dashboard] Totals file not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_spread(pred_date: date) -> pd.DataFrame:
    """
    Load spread predictions for a given date.

    Expected path pattern:
      data/predictions_spread/spread_{date}.parquet
    """
    path = SPREAD_PRED_DIR / f"spread_{pred_date}.parquet"
    if not path.exists():
        logger.warning(f"[Dashboard] Spread file not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_combined(pred_date: date) -> pd.DataFrame:
    """
    Load combined predictions.

    v4 orchestrator saves:
      data/predictions_combined/combined_{date}.parquet
    """
    path = COMBINED_PRED_DIR / f"combined_{pred_date}.parquet"
    if not path.exists():
        logger.warning(f"[Dashboard] Combined predictions not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


# ------------------------------------------------------------
# Combine predictions (fallback if combined file missing)
# ------------------------------------------------------------


def _combine_predictions(pred_date: date) -> pd.DataFrame:
    """
    Try to load pre-merged combined predictions; if not found,
    reconstruct from individual prediction files where possible.
    """
    combined = _load_combined(pred_date)
    if not combined.empty:
        logger.info("[Dashboard] Using pre-combined predictions file.")
        return combined

    # Fallback: manually join individual predictions
    ml = _load_moneyline(pred_date)
    totals = _load_totals(pred_date)
    spread = _load_spread(pred_date)

    if ml.empty and totals.empty and spread.empty:
        logger.warning("[Dashboard] No individual prediction files found to combine.")
        return pd.DataFrame()

    # Moneyline: collapse home/away rows
    if not ml.empty:
        if "is_home" not in ml.columns:
            logger.error(
                "[Dashboard] Moneyline predictions missing 'is_home' column; "
                "cannot pivot to home/away rows."
            )
            ml_combined = pd.DataFrame()
        else:
            ml_home = ml[ml["is_home"] == 1].rename(
                columns={"team": "home_team", "opponent": "away_team"}
            )
            ml_away = ml[ml["is_home"] == 0].rename(
                columns={"team": "away_team", "opponent": "home_team"}
            )

            ml_home = ml_home[
                ["game_id", "date", "home_team", "away_team", "win_probability"]
            ].rename(columns={"win_probability": "win_probability_home"})

            ml_away = ml_away[["game_id", "win_probability"]].rename(
                columns={"win_probability": "win_probability_away"}
            )

            ml_combined = ml_home.merge(ml_away, on="game_id", how="left")
    else:
        ml_combined = pd.DataFrame()

    # Totals
    if not totals.empty:
        totals_trim = totals[
            ["game_id", "home_team", "away_team", "predicted_total_points"]
        ]
    else:
        totals_trim = pd.DataFrame()

    # Spread
    if not spread.empty:
        spread_trim = spread[["game_id", "home_team", "away_team", "predicted_margin"]]
    else:
        spread_trim = pd.DataFrame()

    # Merge everything
    combined = ml_combined
    if combined.empty:
        if not totals_trim.empty:
            combined = totals_trim.copy()
        elif not spread_trim.empty:
            combined = spread_trim.copy()

    if combined.empty:
        logger.warning("[Dashboard] Could not create any combined DataFrame.")
        return pd.DataFrame()

    if not totals_trim.empty:
        combined = combined.merge(
            totals_trim,
            on=["game_id", "home_team", "away_team"],
            how="left",
            suffixes=("", "_totals"),
        )

    if not spread_trim.empty:
        combined = combined.merge(
            spread_trim,
            on=["game_id", "home_team", "away_team"],
            how="left",
            suffixes=("", "_spread"),
        )

    return combined


def get_today() -> date:
    return date.today()


# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------


def main():
    st.set_page_config(page_title="NBA Predictions Dashboard", layout="wide")

    st.title("🏀 NBA Predictions Dashboard (v4)")
    st.markdown(
        "Explore daily model outputs for moneyline, totals, and spread predictions."
    )

    # Sidebar controls
    today = date.today()
    pred_date = st.sidebar.date_input("Prediction date", value=today)
    if isinstance(pred_date, str):
        pred_date = date.fromisoformat(pred_date)

    team_filter = st.sidebar.text_input(
        "Filter by team (tricode or name; optional)",
        value="",
        help="Partial match on home or away team.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Powered by NBA Analytics v4")

    # Load predictions
    combined = _combine_predictions(pred_date)

    if combined.empty:
        st.warning(f"No predictions found for {pred_date}.")
        return

    # Apply team filter
    if team_filter.strip():
        t = team_filter.strip().lower()
        combined = combined[
            combined["home_team"].astype(str).str.lower().str.contains(t)
            | combined["away_team"].astype(str).str.lower().str.contains(t)
        ]

    if combined.empty:
        st.warning("No games match the current filter.")
        return

    # Model versions
    if "model_version" in combined.columns:
        versions = sorted(combined["model_version"].dropna().unique().tolist())
        st.caption(f"Model versions used: {versions}")
    else:
        st.caption("Model version info unavailable.")

    # Summary metrics
    st.subheader(f"Games on {pred_date}")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Games", len(combined))

    with col2:
        if "win_probability_home" in combined.columns:
            fav = combined["win_probability_home"].max()
            st.metric("Highest home win prob", f"{fav:.1%}")
        else:
            st.metric("Highest home win prob", "N/A")

    with col3:
        if "predicted_total_points" in combined.columns:
            max_total = combined["predicted_total_points"].max()
            st.metric("Max predicted total", f"{max_total:.1f}")
        else:
            st.metric("Max predicted total", "N/A")

    with col4:
        if "predicted_margin" in combined.columns:
            max_abs_margin = combined["predicted_margin"].abs().max()
            st.metric("Largest edge (abs margin)", f"{max_abs_margin:.1f}")
        else:
            st.metric("Largest edge", "N/A")

    # Detailed table
    st.subheader("Game-level predictions")

    display_cols = [
        "game_id",
        "home_team",
        "away_team",
        "win_probability_home",
        "win_probability_away",
        "predicted_total_points",
        "predicted_margin",
    ]
    display_cols = [c for c in display_cols if c in combined.columns]

    st.dataframe(
        combined[display_cols].sort_values("game_id"),
        use_container_width=True,
    )

    # Optional: edges
    edge_cols = [c for c in ["home_edge", "away_edge"] if c in combined.columns]
    if edge_cols:
        st.subheader("Edges vs Market")
        edges_view = combined[["game_id", "home_team", "away_team"] + edge_cols].copy()
        for c in edge_cols:
            edges_view[c] = edges_view[c] * 100.0
        st.dataframe(edges_view.sort_values(edge_cols, ascending=False))

    # Raw data
    with st.expander("Show raw prediction data"):
        st.dataframe(combined, use_container_width=True)


if __name__ == "__main__":
    main()
