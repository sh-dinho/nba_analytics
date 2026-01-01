from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Feature Schema
# File: src/features/feature_schema.py
# Author: Sadiq
#
# Description:
#     Canonical schema for model-ready feature rows.
# ============================================================

from typing import Optional
from pydantic import BaseModel, validator


class FeatureRow(BaseModel):
    # --------------------------------------------------------
    # Identifiers
    # --------------------------------------------------------
    game_id: str
    team: str
    opponent: str
    season: str

    # --------------------------------------------------------
    # Core features
    # --------------------------------------------------------
    is_home_feature: int
    team_win_pct_last10: Optional[float]
    opp_win_pct_last10: Optional[float]

    # --------------------------------------------------------
    # ELO features
    # --------------------------------------------------------
    elo: Optional[float]
    opp_elo: Optional[float]
    elo_roll5: Optional[float]
    elo_roll10: Optional[float]

    # --------------------------------------------------------
    # Rolling margin + win rate
    # --------------------------------------------------------
    margin_last5: Optional[float]
    margin_last10: Optional[float]
    roll_win_rate_5: Optional[float]
    roll_win_rate_10: Optional[float]

    # --------------------------------------------------------
    # Form
    # --------------------------------------------------------
    form_last3: Optional[float]
    win_streak: Optional[int]

    # --------------------------------------------------------
    # Rest
    # --------------------------------------------------------
    rest_days: Optional[int]
    is_b2b: Optional[int]

    # --------------------------------------------------------
    # Strength of schedule
    # --------------------------------------------------------
    sos: Optional[float]

    # --------------------------------------------------------
    # Validators
    # --------------------------------------------------------
    @validator("game_id", "team", "opponent")
    def validate_non_empty(cls, v):
        if v is None or str(v).strip() == "":
            raise ValueError("Identifier fields must be non-empty")
        return v

    @validator("season")
    def validate_season_format(cls, v):
        if not isinstance(v, str) or v.count("-") != 1:
            raise ValueError(f"Invalid season format: {v}")
        return v

    @validator("is_home_feature", "is_b2b")
    def validate_binary(cls, v):
        if v is not None and v not in (0, 1):
            raise ValueError(f"Expected binary 0/1, got {v}")
        return v

    @validator("team_win_pct_last10", "opp_win_pct_last10",
               "roll_win_rate_5", "roll_win_rate_10")
    def validate_pct(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"Percentage must be in [0,1], got {v}")
        return v

    @validator("elo", "opp_elo", "elo_roll5", "elo_roll10")
    def validate_elo(cls, v):
        if v is not None and not (500 <= v <= 3000):
            raise ValueError(f"ELO value out of expected range: {v}")
        return v

    @validator("rest_days")
    def validate_rest_days(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"rest_days cannot be negative, got {v}")
        return v