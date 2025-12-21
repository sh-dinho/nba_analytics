# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Build value-bet candidates by comparing model
#              win probabilities against market moneyline odds.
# ============================================================

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ------------------------------------------------------------
# Moneyline conversions (safe, vectorized-friendly)
# ------------------------------------------------------------


def _parse_ml(ml: object) -> Optional[int]:
    """
    Safely parse a moneyline value into an int.
    Returns None for malformed or non-numeric values.
    """
    if ml is None:
        return None
    if isinstance(ml, (int, np.integer)):
        return int(ml)
    try:
        s = str(ml).strip().upper()
        # Common non-priced markers
        if s in {"EVEN", "PK", "OFF", "NA", "", "—", "-", "NONE"}:
            return None
        return int(s)
    except Exception:
        return None


def ml_to_implied_prob(ml: object) -> Optional[float]:
    """
    Convert American moneyline to implied probability.
    Returns None if ml cannot be parsed or is invalid.
    """
    parsed = _parse_ml(ml)
    if parsed is None:
        return None

    if parsed > 0:
        return 100.0 / (parsed + 100.0)
    else:
        return -parsed / (-parsed + 100.0)


def ml_to_decimal_odds(ml: object) -> Optional[float]:
    """
    Convert American moneyline to decimal odds.
    Returns None if ml cannot be parsed or is invalid.
    """
    parsed = _parse_ml(ml)
    if parsed is None:
        return None

    if parsed > 0:
        return 1.0 + parsed / 100.0
    else:
        return 1.0 + 100.0 / float(-parsed)


# ------------------------------------------------------------
# Core value-bet builder
# ------------------------------------------------------------


def build_value_bets(preds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """
    Build value-bet candidates by merging model predictions with market odds.

    Inputs:
        preds: long-format predictions at team level, expected columns:
            - game_id
            - team
            - is_home
            - win_probability
            - prediction_date
            - model_version
            - feature_version

        odds: odds snapshot (e.g. from ESPN), expected columns:
            - game_id
            - home_team
            - away_team
            - home_ml
            - away_ml

    Output:
        DataFrame where each row is a side (home/away) of a game, with:
            - game_id
            - market_team
            - market_side ("home" / "away")
            - ml
            - model_prob
            - implied_prob
            - decimal_odds
            - edge
            - ev_per_unit
            - kelly_fraction
            - prediction_date
            - model_version
            - feature_version
    """
    # --------------------------------------------------------
    # Early exits and schema validation
    # --------------------------------------------------------
    if preds is None or preds.empty:
        logger.info("build_value_bets(): empty preds; returning empty DataFrame.")
        return pd.DataFrame()

    if odds is None or odds.empty:
        logger.info("build_value_bets(): empty odds; returning empty DataFrame.")
        return pd.DataFrame()

    required_preds = {
        "game_id",
        "team",
        "is_home",
        "win_probability",
        "prediction_date",
        "model_version",
        "feature_version",
    }
    required_odds = {
        "game_id",
        "home_team",
        "away_team",
        "home_ml",
        "away_ml",
    }

    missing_preds = required_preds - set(preds.columns)
    missing_odds = required_odds - set(odds.columns)

    if missing_preds:
        raise ValueError(f"build_value_bets(): missing preds columns: {missing_preds}")
    if missing_odds:
        raise ValueError(f"build_value_bets(): missing odds columns: {missing_odds}")

    # --------------------------------------------------------
    # Merge predictions with odds
    # --------------------------------------------------------
    # We assume game_id alignment; team names are used only for labeling.
    df = preds.merge(odds, on="game_id", how="inner")
    if df.empty:
        logger.warning(
            "build_value_bets(): merge produced no rows; check game_id alignment."
        )
        return pd.DataFrame()

    # --------------------------------------------------------
    # Split into home/away sides
    # --------------------------------------------------------
    home = df[df["is_home"] == 1].copy()
    home["market_team"] = home["home_team"]
    home["ml"] = home["home_ml"]
    home["market_side"] = "home"

    away = df[df["is_home"] == 0].copy()
    away["market_team"] = away["away_team"]
    away["ml"] = away["away_ml"]
    away["market_side"] = "away"

    bets = pd.concat([home, away], ignore_index=True)

    if bets.empty:
        logger.warning(
            "build_value_bets(): no home/away rows after split; returning empty."
        )
        return bets

    # --------------------------------------------------------
    # Compute implied probabilities and decimal odds (safe)
    # --------------------------------------------------------
    bets["implied_prob"] = bets["ml"].apply(ml_to_implied_prob)
    bets["decimal_odds"] = bets["ml"].apply(ml_to_decimal_odds)

    # Model probabilities
    bets["model_prob"] = bets["win_probability"]

    # Drop rows with invalid probs/odds
    before = len(bets)
    bets = bets[
        bets["implied_prob"].between(0.0, 1.0, inclusive="both")
        & bets["model_prob"].between(0.0, 1.0, inclusive="both")
        & bets["decimal_odds"].notna()
    ].copy()
    dropped = before - len(bets)
    if dropped:
        logger.warning(
            f"build_value_bets(): dropped {dropped} rows with invalid probs/odds."
        )

    if bets.empty:
        logger.warning(
            "build_value_bets(): all rows dropped after validation; returning empty."
        )
        return bets

    # --------------------------------------------------------
    # Edge and EV per unit
    # --------------------------------------------------------
    bets["edge"] = bets["model_prob"] - bets["implied_prob"]

    b = bets["decimal_odds"] - 1.0
    p = bets["model_prob"]
    q = 1.0 - p

    bets["ev_per_unit"] = p * b - q

    # --------------------------------------------------------
    # Kelly fraction (safe, no division by zero, no negative)
    # --------------------------------------------------------
    # Kelly formula: f* = (b*p - q) / b, where b = decimal_odds - 1
    kelly_raw = (b * p - q) / b.replace(0, np.nan)
    # Replace infinities/NaNs and negatives with 0 (no-bet)
    kelly_clean = kelly_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    kelly_clean = kelly_clean.clip(lower=0.0)

    bets["kelly_fraction"] = kelly_clean

    # --------------------------------------------------------
    # Final column selection and ordering
    # --------------------------------------------------------
    out_cols = [
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

    # Only keep columns that exist (defensive)
    out_cols = [c for c in out_cols if c in bets.columns]

    bets = bets[out_cols].copy()

    logger.info(
        f"build_value_bets(): produced {len(bets)} candidate bets "
        f"from {len(df)} merged prediction+odds rows."
    )

    return bets
