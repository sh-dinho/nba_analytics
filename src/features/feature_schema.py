from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Feature Schema
# File: src/features/feature_schema.py
# Author: Sadiq
#
# Description:
#     Canonical schema for model-ready feature rows.
#     Updated to allow early-season NaNs and match pipeline output.
# ============================================================

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, field_validator
import pandas as pd


class FeatureRow(BaseModel):
    # --------------------------------------------------------
    # Identifiers
    # --------------------------------------------------------
    game_id: str
    team: str
    opponent: str
    season: str
    date: datetime
    is_home: int

    # --------------------------------------------------------
    # Core stats
    # --------------------------------------------------------
    score: int
    opp_score: int
    win: int
    margin: int
    total_points: int

    # --------------------------------------------------------
    # ELO features
    # --------------------------------------------------------
    elo: Optional[float]
    opp_elo: Optional[float]
    elo_roll5: Optional[float]
    elo_roll10: Optional[float]

    # --------------------------------------------------------
    # Rolling stats
    # --------------------------------------------------------
    points_for_rolling_5: Optional[float]
    points_against_rolling_5: Optional[float]
    margin_rolling_5: Optional[float]
    win_rolling_5: Optional[float]

    points_for_rolling_10: Optional[float]
    points_against_rolling_10: Optional[float]
    margin_rolling_10: Optional[float]

    points_for_rolling_20: Optional[float]
    points_against_rolling_20: Optional[float]
    margin_rolling_20: Optional[float]
    win_rolling_20: Optional[float]

    # --------------------------------------------------------
    # Contextual features
    # --------------------------------------------------------
    rest_days: Optional[int]
    is_b2b: Optional[int]
    form_last3: Optional[float]
    sos: Optional[float]
    win_streak: Optional[int]

    # --------------------------------------------------------
    # Opponent-adjusted features
    # --------------------------------------------------------
    opp_margin_rolling_5: Optional[float]
    opp_margin_rolling_10: Optional[float]
    team_win_pct_last10: Optional[float]
    opp_win_pct_last10: Optional[float]

    # --------------------------------------------------------
    # Validators
    # --------------------------------------------------------
    @field_validator("game_id", "team", "opponent", "season")
    def validate_non_empty(cls, v):
        if not v or str(v).strip() == "":
            raise ValueError("Identifier fields must be non-empty")
        return v

    @field_validator("is_home", "is_b2b", "win")
    def validate_binary(cls, v):
        if v not in (0, 1):
            raise ValueError(f"Expected binary 0/1, got {v}")
        return v

    @field_validator(
        "win_rolling_5",
        "win_rolling_20",
        "team_win_pct_last10",
        "opp_win_pct_last10",
    )
    def validate_win_rate(cls, v):
        if v is None or pd.isna(v):
            return v
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Win rate must be in [0,1], got {v}")
        return v

    @field_validator("elo", "opp_elo", "elo_roll5", "elo_roll10")
    def validate_elo(cls, v):
        if v is None or pd.isna(v):
            return v
        if not (500 <= v <= 3000):
            raise ValueError(f"ELO value out of expected range: {v}")
        return v

    @field_validator("rest_days")
    def validate_rest_days(cls, v):
        if v is None or pd.isna(v):
            return v
        if v < 0:
            raise ValueError(f"rest_days cannot be negative, got {v}")
        return v

    @field_validator("sos")
    def validate_sos(cls, v):
        if v is None or pd.isna(v):
            return v
        if not (-200 <= v <= 200):
            raise ValueError(f"SOS value out of expected range: {v}")
        return v