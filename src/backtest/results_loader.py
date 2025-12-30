# ============================================================
# 🏀 NBA Analytics v4
# Module: Backtest Results Loader
# File: src/backtest/results_loader.py
# Author: Sadiq
#
# Description:
#     Loads actual game results + closing odds for backtesting.
#     Expected schema:
#         game_id
#         date
#         home_team
#         away_team
#         home_score
#         away_score
#         closing_home_odds
#         closing_away_odds
# ============================================================

from __future__ import annotations

import pandas as pd
from pathlib import Path
from loguru import logger

from src.config.paths import DATA_DIR


RESULTS_PATH = DATA_DIR / "results" / "results.parquet"


def load_results() -> pd.DataFrame:
    """
    Load historical results + closing odds for backtesting.

    Returns:
        pd.DataFrame with required columns.
    """
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Backtest results file not found at {RESULTS_PATH}. "
            "Create results.parquet with final scores + closing odds."
        )

    df = pd.read_parquet(RESULTS_PATH)

    required = {
        "game_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "closing_home_odds",
        "closing_away_odds",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results file missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"]).dt.date

    logger.info(f"[Backtest] Loaded results: {len(df)} rows from {RESULTS_PATH}")

    return df
