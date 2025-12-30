# ============================================================
# 🏀 NBA Analytics v4
# Module: Prediction Edges
# File: src/model/edges.py
# Author: Sadiq
#
# Description:
#     Computes implied odds and edges vs market for
#     moneyline, totals, and spread predictions.
# ============================================================

from __future__ import annotations

import pandas as pd


def _prob_to_fair_decimal_odds(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 1.0 / p


def add_moneyline_edges(pred_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    pred_df: one row per game, with win_probability_home, win_probability_away
    market_df: one row per game, with columns:
        game_id, home_ml_odds, away_ml_odds (decimal odds)
    """
    df = pred_df.merge(market_df, on="game_id", how="left")

    # Fair odds
    df["fair_home_odds"] = df["win_probability_home"].apply(_prob_to_fair_decimal_odds)
    df["fair_away_odds"] = df["win_probability_away"].apply(_prob_to_fair_decimal_odds)

    # Edges in percentage terms: (market_implied_prob - model_prob)
    df["home_edge"] = (1 / df["home_ml_odds"]) - df["win_probability_home"]
    df["away_edge"] = (1 / df["away_ml_odds"]) - df["win_probability_away"]

    return df
