from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Strength of Schedule Features
# File: src/features/sos.py
# Author: Sadiq
#
# Description:
#     Strength-of-schedule proxy based on opponent defensive
#     performance (rolling points allowed).
# ============================================================

import pandas as pd
from loguru import logger


def add_sos_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds strength-of-schedule (SOS) proxies.

    Calculation:
    1. Compute each team's rolling points allowed (opp_score) over last 10 games.
    2. Shift by 1 to ensure leakage-safe pre-game knowledge.
    3. Map opponent's defensive profile into each row using game_id.
    """
    out = df.copy()

    # Validate required columns
    required = {"team", "opponent", "date", "game_id", "opp_score"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"add_sos_features missing columns: {missing}")

    # Ensure datetime
    out["date"] = pd.to_datetime(out["date"])

    # Sort for rolling operations
    out = out.sort_values(["team", "date"]).reset_index(drop=True)

    # 1. Rolling defensive profile (points allowed)
    out["opp_points_allowed_roll10"] = (
        out.groupby("team")["opp_score"]
        .shift(1)
        .rolling(10, min_periods=1)
        .mean()
        .astype("float32")
    )

    # 2. Build lookup table for opponent's defensive profile
    sos_lookup = (
        out[["game_id", "team", "opp_points_allowed_roll10"]]
        .rename(columns={
            "team": "opponent",
            "opp_points_allowed_roll10": "sos"
        })
    )

    # 3. Merge opponent SOS into each row
    out = out.merge(sos_lookup, on=["game_id", "opponent"], how="left")

    # Fill early-season NaNs with neutral league average
    out["sos"] = out["sos"].fillna(112.0).astype("float32")

    # Drop intermediate column
    out = out.drop(columns=["opp_points_allowed_roll10"])

    logger.debug("Added SOS (defensive difficulty) features.")
    return out
