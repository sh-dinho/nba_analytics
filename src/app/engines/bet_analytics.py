from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v5.0
# Module: Bet Analytics Engine
# Purpose:
#     Core analytics for logged bets:
#       • portfolio-level summaries
#       • breakdowns by market / team / tag
#       • edge calibration
#       • parlay aggregation
# ============================================================

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ------------------------------------------------------------
# Summary Dataclass
# ------------------------------------------------------------

@dataclass(frozen=True)
class BetSummary:
    total_bets: int
    total_stake: float
    total_pnl: float
    win_rate: Optional[float]
    avg_stake: Optional[float]
    roi: Optional[float]  # pnl / stake


# ------------------------------------------------------------
# Core Portfolio Summary
# ------------------------------------------------------------

def summarize_bets(df: pd.DataFrame) -> BetSummary:
    if df.empty:
        return BetSummary(
            total_bets=0,
            total_stake=0.0,
            total_pnl=0.0,
            win_rate=None,
            avg_stake=None,
            roi=None,
        )

    total_bets = len(df)
    total_stake = float(df["stake"].sum()) if "stake" in df.columns else 0.0
    total_pnl = float(df["pnl"].sum()) if "pnl" in df.columns else 0.0

    win_rate: Optional[float] = None
    if "result" in df.columns:
        wins = (df["result"] == "win").sum()
        win_rate = wins / total_bets if total_bets > 0 else None

    avg_stake: Optional[float] = total_stake / total_bets if total_bets > 0 else None
    roi: Optional[float] = total_pnl / total_stake if total_stake > 0 else None

    return BetSummary(
        total_bets=total_bets,
        total_stake=total_stake,
        total_pnl=total_pnl,
        win_rate=win_rate,
        avg_stake=avg_stake,
        roi=roi,
    )


# ------------------------------------------------------------
# Generic Breakdown Helper
# ------------------------------------------------------------

def breakdown_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Group bets by a column (e.g., 'market', 'team', 'tag') and compute:
      - bets, stake, pnl, wins, win_rate, roi
    Expects: bet_id, stake, pnl, result.
    """
    required = {"bet_id", "stake", "pnl", "result"}
    if df.empty or not required.issubset(df.columns) or group_col not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            bets=("bet_id", "count"),
            stake=("stake", "sum"),
            pnl=("pnl", "sum"),
            wins=("result", lambda s: (s == "win").sum()),
        )
        .reset_index()
    )

    grouped["roi"] = grouped["pnl"] / grouped["stake"].replace(0, pd.NA)
    grouped["win_rate"] = grouped["wins"] / grouped["bets"].replace(0, pd.NA)

    return grouped


# ------------------------------------------------------------
# Edge / Probability Calibration
# ------------------------------------------------------------

def edge_calibration(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Bin bets by model_prob and compare expected vs actual.
    Requires: model_prob, result, bet_id.
    """
    required = {"bet_id", "model_prob", "result"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    df = df.copy()
    df = df[df["result"].isin(["win", "loss"])]

    if df.empty:
        return pd.DataFrame()

    df["bin"] = pd.qcut(df["model_prob"], q=n_bins, duplicates="drop")

    grouped = (
        df.groupby("bin")
        .agg(
            bets=("bet_id", "count"),
            avg_model_prob=("model_prob", "mean"),
            actual_win_rate=("result", lambda s: (s == "win").mean()),
        )
        .reset_index()
    )

    grouped["calibration_gap"] = grouped["actual_win_rate"] - grouped["avg_model_prob"]

    return grouped


# ------------------------------------------------------------
# Parlay Aggregation
# ------------------------------------------------------------

def aggregate_parlays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate parlay legs into parlay-level metrics.
    Requires: parlay_group_id, stake, pnl, placed_at (optional).
    """
    if df.empty or "parlay_group_id" not in df.columns:
        return pd.DataFrame()

    base = df.dropna(subset=["parlay_group_id"]).copy()
    if base.empty:
        return pd.DataFrame()

    agg = (
        base.groupby("parlay_group_id")
        .agg(
            legs=("bet_id", "count"),
            stake=("stake", "sum"),
            pnl=("pnl", "sum"),
            placed_at=("placed_at", "min"),
        )
        .reset_index()
    )

    agg["roi"] = agg["pnl"] / agg["stake"].replace(0, pd.NA)

    return agg
