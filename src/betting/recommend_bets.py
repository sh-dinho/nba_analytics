from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Recommendation Engine
# File: src/betting/recommend_bets.py
# Author: Sadiq
#
# Description:
#     Rank value bets into bankroll‑ and risk‑aware betting
#     recommendations with confidence scoring and exposure limits.
# ============================================================

import pandas as pd
from loguru import logger

from src.config.betting import (
    BANKROLL,
    KELLY_CAP,
    KELLY_MIN,
    MAX_TOTAL_EXPOSURE,
    MAX_EXPOSURE_PER_GAME,
    MAX_BETS_PER_DAY,
    CONFIDENCE_WEIGHTS,
)
from src.config.thresholds import MIN_EDGE, MIN_EV


REQUIRED_COLUMNS = [
    "game_id",
    "market_team",
    "market_side",
    "ml",
    "model_prob",
    "implied_prob",
    "edge",
    "decimal_odds",
    "ev_per_unit",
    "kelly_fraction",
    "prediction_date",
    "model_version",
    "feature_version",
]


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
def _validate(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        logger.info("recommend_bets(): received empty value‑bet DataFrame.")
        return pd.DataFrame()

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"recommend_bets(): missing required columns: {missing}")

    df = df[
        df["model_prob"].between(0, 1)
        & df["implied_prob"].between(0, 1)
        & df["decimal_odds"].gt(1.0)
        & df["kelly_fraction"].ge(0.0)
    ].copy()

    if df.empty:
        logger.warning("recommend_bets(): all rows dropped after validation.")

    return df


# ------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------
def _apply_thresholds(df: pd.DataFrame, min_edge: float, min_ev: float) -> pd.DataFrame:
    before = len(df)
    df = df[(df["edge"] >= min_edge) & (df["ev_per_unit"] >= min_ev)].copy()
    dropped = before - len(df)

    if dropped:
        logger.info(
            f"recommend_bets(): dropped {dropped} bets below thresholds "
            f"(min_edge={min_edge}, min_ev={min_ev})."
        )

    return df


# ------------------------------------------------------------
# Kelly Caps
# ------------------------------------------------------------
def _apply_kelly_caps(df: pd.DataFrame, max_kelly: float) -> pd.DataFrame:
    df = df.copy()
    df["kelly_fraction"] = df["kelly_fraction"].clip(lower=KELLY_MIN, upper=max_kelly)
    return df


# ------------------------------------------------------------
# Stake Sizing
# ------------------------------------------------------------
def _compute_stakes(df: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    df = df.copy()
    df["recommended_stake"] = bankroll * df["kelly_fraction"]
    df = df[df["recommended_stake"] > 0].copy()
    return df


# ------------------------------------------------------------
# Confidence Scoring
# ------------------------------------------------------------
def _compute_confidence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    w_edge = CONFIDENCE_WEIGHTS["edge"]
    w_ev = CONFIDENCE_WEIGHTS["ev"]
    w_kelly = CONFIDENCE_WEIGHTS["kelly"]

    def scale(col: pd.Series) -> pd.Series:
        c = col.fillna(0)
        c_min, c_max = c.min(), c.max()
        if c_min == c_max:
            return pd.Series(0.5, index=c.index)
        return (c - c_min) / (c_max - c_min)

    df["confidence"] = (
        w_edge * scale(df["edge"])
        + w_ev * scale(df["ev_per_unit"])
        + w_kelly * scale(df["kelly_fraction"])
    ) * 100

    return df


# ------------------------------------------------------------
# Exposure Limits
# ------------------------------------------------------------
def _apply_exposure_limits(df: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    if df.empty:
        return df

    max_game = bankroll * MAX_EXPOSURE_PER_GAME
    max_total = bankroll * MAX_TOTAL_EXPOSURE

    selected = []
    total = 0.0
    per_game: dict[str, float] = {}

    for idx, row in df.iterrows():
        stake = float(row["recommended_stake"])
        game = row["game_id"]

        if stake <= 0:
            continue

        # Per‑game exposure
        if per_game.get(game, 0.0) + stake > max_game:
            continue

        # Total exposure
        if total + stake > max_total:
            break

        selected.append(idx)
        total += stake
        per_game[game] = per_game.get(game, 0.0) + stake

        if len(selected) >= MAX_BETS_PER_DAY:
            break

    return df.loc[selected].copy()


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def recommend_bets(
    bets_df: pd.DataFrame,
    bankroll: float = BANKROLL,
    min_edge: float = MIN_EDGE,
    min_ev: float = MIN_EV,
    max_kelly: float = KELLY_CAP,
) -> pd.DataFrame:
    """
    Produce ranked, bankroll‑aware betting recommendations.

    Steps:
        1. Validate input
        2. Apply edge + EV thresholds
        3. Apply Kelly caps
        4. Compute stake sizes
        5. Compute confidence score
        6. Apply exposure limits
    """
    df = _validate(bets_df)
    if df.empty:
        return df

    df = _apply_thresholds(df, min_edge, min_ev)
    if df.empty:
        return df

    df = _apply_kelly_caps(df, max_kelly)
    df = _compute_stakes(df, bankroll)
    if df.empty:
        return df

    df = _compute_confidence(df)
    df = df.sort_values("confidence", ascending=False)

    df = _apply_exposure_limits(df, bankroll)
    return df