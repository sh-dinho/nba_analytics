from __future__ import annotations
# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Rank value bets into bankroll- and risk-aware
#              betting recommendations with confidence scoring.
# ============================================================


from typing import List

import numpy as pd
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


REQUIRED_VALUE_BET_COLUMNS: List[str] = [
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


def _validate_input(bets_df: pd.DataFrame) -> pd.DataFrame:
    if bets_df is None or bets_df.empty:
        logger.info("recommend_bets(): received empty bets_df.")
        return pd.DataFrame()

    missing = set(REQUIRED_VALUE_BET_COLUMNS) - set(bets_df.columns)
    if missing:
        raise ValueError(f"recommend_bets(): missing required columns: {missing}")

    df = bets_df.copy()

    # Basic sanity checks
    df = df[
        df["model_prob"].between(0.0, 1.0, inclusive="both")
        & df["implied_prob"].between(0.0, 1.0, inclusive="both")
        & df["decimal_odds"].gt(1.0)
        & df["kelly_fraction"].ge(0.0)
    ].copy()

    if df.empty:
        logger.warning("recommend_bets(): all rows dropped after validation.")
        return df

    return df


def _compute_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a normalized confidence score between 0 and 100.
    Uses weighted, scaled versions of edge, EV, and Kelly.
    """

    w_edge = CONFIDENCE_WEIGHTS.get("edge", 0.5)
    w_ev = CONFIDENCE_WEIGHTS.get("ev", 0.3)
    w_kelly = CONFIDENCE_WEIGHTS.get("kelly", 0.2)

    def _scale(col: pd.Series) -> pd.Series:
        # Min-max scaling with safe fallback
        c = col.fillna(0.0)
        c_min, c_max = c.min(), c.max()
        if c_max == c_min:
            return pd.Series(0.5, index=c.index)  # neutral score if no variance
        return (c - c_min) / (c_max - c_min)

    edge_scaled = _scale(df["edge"])
    ev_scaled = _scale(df["ev_per_unit"])
    kelly_scaled = _scale(df["kelly_fraction"])

    confidence = (
        w_edge * edge_scaled + w_ev * ev_scaled + w_kelly * kelly_scaled
    ) * 100.0

    df["confidence"] = confidence
    return df


def _apply_kelly_caps(
    df: pd.DataFrame, kelly_cap: float, kelly_min: float
) -> pd.DataFrame:
    df["kelly_fraction"] = df["kelly_fraction"].clip(lower=kelly_min, upper=kelly_cap)
    return df


def _compute_stakes(df: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    df["recommended_stake"] = bankroll * df["kelly_fraction"]
    return df


def _apply_exposure_limits(df: pd.DataFrame, bankroll: float) -> pd.DataFrame:
    """
    Enforce:
      - max exposure per game
      - max total exposure per day
      - max number of bets per day
    Assumes df is sorted by confidence descending.
    """

    if df.empty:
        return df

    max_per_game = bankroll * MAX_EXPOSURE_PER_GAME
    max_total = bankroll * MAX_TOTAL_EXPOSURE

    selected_rows = []
    total_stake = 0.0
    per_game_stake = {}

    for idx, row in df.iterrows():
        game_id = row["game_id"]
        stake = float(row["recommended_stake"])

        # Skip zero-stake bets
        if stake <= 0:
            continue

        # Per-game limit
        game_stake = per_game_stake.get(game_id, 0.0)
        if game_stake + stake > max_per_game:
            logger.debug(
                f"Skipping bet on game_id={game_id} due to per-game exposure limit. "
                f"proposed={game_stake + stake:.2f}, limit={max_per_game:.2f}"
            )
            continue

        # Total exposure limit
        if total_stake + stake > max_total:
            logger.info(
                f"Stopping selection due to total exposure limit: "
                f"proposed_total={total_stake + stake:.2f}, limit={max_total:.2f}"
            )
            break

        selected_rows.append(idx)
        total_stake += stake
        per_game_stake[game_id] = game_stake + stake

        if len(selected_rows) >= MAX_BETS_PER_DAY:
            logger.info(
                f"Reached MAX_BETS_PER_DAY={MAX_BETS_PER_DAY}; "
                f"stopping further selections."
            )
            break

    if not selected_rows:
        logger.info("No bets selected after applying exposure limits.")
        return df.iloc[0:0]

    return df.loc[selected_rows].copy()


def explain_confidence(row: pd.Series) -> str:
    """
    Human-readable explanation of why a bet is recommended
    and how strong it is.
    """
    reasons = []

    # Edge
    if row["edge"] > 0.10:
        reasons.append(
            "Model probability is significantly higher than the market's implied probability."
        )
    elif row["edge"] > 0.05:
        reasons.append(
            "Model probability is moderately higher than the market's implied probability."
        )
    elif row["edge"] > 0.02:
        reasons.append("Model shows a small but meaningful edge over the market.")
    else:
        reasons.append("Model shows only a marginal advantage over the market.")

    # EV
    if row["ev_per_unit"] > 0.05:
        reasons.append("Expected value per unit staked is strong.")
    elif row["ev_per_unit"] > 0.02:
        reasons.append("Expected value per unit is positive.")
    else:
        reasons.append("Expected value is modest.")

    # Kelly
    if row["kelly_fraction"] > 0.05:
        reasons.append("Kelly suggests a relatively aggressive stake.")
    elif row["kelly_fraction"] > 0.02:
        reasons.append("Kelly suggests a moderate stake size.")
    elif row["kelly_fraction"] > 0.0:
        reasons.append("Kelly suggests a small, conservative stake.")
    else:
        reasons.append("Kelly suggests no bet or an extremely small stake.")

    return " ".join(reasons)


def recommend_bets(
    bets_df: pd.DataFrame,
    bankroll: float = BANKROLL,
    min_edge: float = None,
    min_ev: float = None,
    max_kelly: float = None,
) -> pd.DataFrame:
    """
    Generate ranked, risk-aware betting recommendations from value bets.

    Inputs:
        bets_df: output from build_value_bets(), with at least REQUIRED_VALUE_BET_COLUMNS.
        bankroll: current bankroll to size bets against.
        min_edge: optional minimum edge filter (defaults to config threshold if None).
        min_ev: optional minimum EV per unit filter (defaults to config threshold if None).
        max_kelly: optional Kelly cap override (defaults to KELLY_CAP if None).

    Output:
        DataFrame with recommended bets, sorted by confidence desc, including:
            - recommended_stake
            - confidence
            - (all original value-bet columns)
    """
    from src.config.thresholds import MIN_EDGE, MIN_EV  # avoid circular import at top

    df = _validate_input(bets_df)
    if df.empty:
        return df

    # --------------------------------------------------------
    # Filtering by edge / EV
    # --------------------------------------------------------
    min_edge = MIN_EDGE if min_edge is None else min_edge
    min_ev = MIN_EV if min_ev is None else min_ev
    max_kelly = KELLY_CAP if max_kelly is None else max_kelly

    before = len(df)
    df = df[(df["edge"] >= min_edge) & (df["ev_per_unit"] >= min_ev)].copy()
    dropped = before - len(df)
    if dropped:
        logger.info(
            f"recommend_bets(): filtered out {dropped} bets below edge/EV thresholds "
            f"(min_edge={min_edge}, min_ev={min_ev})."
        )

    if df.empty:
        logger.info("recommend_bets(): no bets remain after edge/EV filtering.")
        return df

    # --------------------------------------------------------
    # Kelly caps and stakes
    # --------------------------------------------------------
    df = _apply_kelly_caps(df, kelly_cap=max_kelly, kelly_min=KELLY_MIN)
    df = _compute_stakes(df, bankroll=bankroll)

    # Drop zero-stake bets
    df = df[df["recommended_stake"] > 0].copy()
    if df.empty:
        logger.info("recommend_bets(): no positive-stake bets after Kelly caps.")
        return df

    # --------------------------------------------------------
    # Confidence scoring
    # --------------------------------------------------------
    df = _compute_confidence(df)

    # Sort by confidence descending
    df = df.sort_values("confidence", ascending=False)

    # --------------------------------------------------------
    # Apply exposure limits
    # --------------------------------------------------------
    df_limited = _apply_exposure_limits(df, bankroll=bankroll)

    if df_limited.empty:
        logger.info("recommend_bets(): no bets selected after exposure limits.")
        return df_limited

    # Optionally add an explanation column
    df_limited["confidence_explanation"] = df_limited.apply(explain_confidence, axis=1)

    # Final ordering of columns
    ordered_cols = [
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
        "recommended_stake",
        "confidence",
        "confidence_explanation",
        "prediction_date",
        "model_version",
        "feature_version",
    ]
    ordered_cols = [c for c in ordered_cols if c in df_limited.columns]

    df_limited = df_limited[ordered_cols]

    logger.info(
        f"recommend_bets(): selected {len(df_limited)} bets with total stake="
        f"{df_limited['recommended_stake'].sum():.2f} "
        f"from bankroll={bankroll:.2f}."
    )

    return df_limited
