# ============================================================
# 🏀 NBA Analytics v4
# Module: Backtesting Engine
# File: src/backtesting/engine.py
# Author: Sadiq
#
# Description:
#     Evaluates historical predictions vs actual results.
# ============================================================

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestResult:
    roi: float
    hit_rate: float
    n_bets: int
    avg_edge: float
    clv: float
    summary: pd.DataFrame


def run_backtest(pred_df: pd.DataFrame, results_df: pd.DataFrame) -> BacktestResult:
    df = pred_df.merge(results_df, on="game_id", how="inner")

    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    df["pick"] = df.apply(
        lambda r: "home" if r["home_edge"] >= r["away_edge"] else "away",
        axis=1,
    )

    df["pick_win"] = df.apply(
        lambda r: (
            1
            if (r["pick"] == "home" and r["home_win"] == 1)
            or (r["pick"] == "away" and r["home_win"] == 0)
            else 0
        ),
        axis=1,
    )

    df["payout"] = df.apply(
        lambda r: (
            (r["closing_home_odds"] - 1)
            if (r["pick"] == "home" and r["pick_win"])
            else (
                (r["closing_away_odds"] - 1)
                if (r["pick"] == "away" and r["pick_win"])
                else -1
            )
        ),
        axis=1,
    )

    roi = df["payout"].mean()
    hit_rate = df["pick_win"].mean()
    avg_edge = df[["home_edge", "away_edge"]].max(axis=1).mean()

    df["clv"] = df.apply(
        lambda r: (
            (1 / r["closing_home_odds"]) - r["win_probability_home"]
            if r["pick"] == "home"
            else (1 / r["closing_away_odds"]) - r["win_probability_away"]
        ),
        axis=1,
    )

    return BacktestResult(
        roi=roi,
        hit_rate=hit_rate,
        n_bets=len(df),
        avg_edge=avg_edge,
        clv=df["clv"].mean(),
        summary=df,
    )
