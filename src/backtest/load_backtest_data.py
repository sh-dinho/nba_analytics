from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Backtest Data Loader
# File: src/backtest/load_backtest_data.py
# Author: Sadiq
#
# Description:
#     Loads predictions + actual results for a date range and
#     merges them into a canonical DataFrame ready for modern
#     backtesting.
# ============================================================

from datetime import datetime
from pathlib import Path
import pandas as pd
from loguru import logger

from src.config.paths import (
    PREDICTIONS_DIR,
    RESULTS_SNAPSHOT_DIR,
)


# ------------------------------------------------------------
# Load predictions
# ------------------------------------------------------------

def _load_predictions(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Loads all prediction parquet files between start_date and end_date.
    Expected filenames:
        predictions_YYYYMMDD.parquet
    """
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()

    dfs = []
    for path in PREDICTIONS_DIR.glob("predictions_*.parquet"):
        try:
            date_str = path.name.split("_")[1].split(".")[0]
            pred_date = datetime.strptime(date_str, "%Y%m%d").date()
        except Exception:
            continue

        if start <= pred_date <= end:
            dfs.append(pd.read_parquet(path))

    if not dfs:
        logger.warning("No prediction files found for date range.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} predictions.")
    return df


# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------

def _load_results(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Loads actual game results from the canonical results snapshot.
    Expected columns:
        game_id, date, home_score, away_score,
        closing_home_odds, closing_away_odds
    """
    df = pd.read_parquet(RESULTS_SNAPSHOT_DIR / "results.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.date

    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()

    df = df[(df["date"] >= start) & (df["date"] <= end)]
    logger.info(f"Loaded {len(df)} results.")
    return df


# ------------------------------------------------------------
# Merge predictions + results
# ------------------------------------------------------------

def merge_predictions_and_results(pred_df: pd.DataFrame, res_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a canonical backtest dataset with:
        - pick
        - win_probability
        - implied probability
        - edge
        - actual outcome
        - closing odds
    """
    if pred_df.empty or res_df.empty:
        return pd.DataFrame()

    df = pred_df.merge(res_df, on="game_id", how="inner")

    # Determine pick
    df["pick"] = df.apply(
        lambda r: r["team"] if r["win_probability"] >= 0.5 else r["opponent"],
        axis=1,
    )

    # Select correct closing odds based on pick
    df["closing_odds"] = df.apply(
        lambda r: r["closing_home_odds"] if r["pick"] == r["team"] else r["closing_away_odds"],
        axis=1,
    )

    # Implied probability
    df["implied_prob"] = 1 / df["closing_odds"]

    # Edge = model probability - implied probability
    df["edge"] = df["win_probability"] - df["implied_prob"]

    # Actual outcome
    df["actual_outcome"] = df.apply(
        lambda r: 1 if (
            (r["pick"] == r["team"] and r["home_score"] > r["away_score"]) or
            (r["pick"] == r["opponent"] and r["away_score"] > r["home_score"])
        ) else 0,
        axis=1,
    )

    return df


# ------------------------------------------------------------
# Public entrypoint
# ------------------------------------------------------------

def load_backtest_data(start_date: str, end_date: str) -> pd.DataFrame:
    logger.info(f"Loading backtest data for {start_date} → {end_date}")

    pred_df = _load_predictions(start_date, end_date)
    res_df = _load_results(start_date, end_date)

    merged = merge_predictions_and_results(pred_df, res_df)

    if merged.empty:
        logger.warning("Merged backtest dataset is empty.")
    else:
        logger.success(f"Merged dataset contains {len(merged)} rows.")

    return merged