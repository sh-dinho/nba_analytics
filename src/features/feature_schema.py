from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics v4
# Module: Feature Schema
# File: src/features/feature_schema.py
# Author: Sadiq
#
# Description:
#     Defines the strict schema for engineered team-game
#     features produced by FeatureBuilder v4 (Option C).
#
#     Ensures:
#       - point-in-time correctness
#       - no leakage
#       - consistent feature types
#       - validation before model training/prediction
# ============================================================

from datetime import date
from typing import Optional
from pydantic import BaseModel, validator


class FeatureRow(BaseModel):
    # Core identifiers
    game_id: str
    team: str
    opponent: str
    date: date
    season: str
    is_home: int  # 0/1

    # Basic features
    score_diff: Optional[float]

    # Rolling stats (5, 10, 20)
    roll_points_for_5: Optional[float]
    roll_points_against_5: Optional[float]
    roll_margin_5: Optional[float]
    roll_win_rate_5: Optional[float]

    roll_points_for_10: Optional[float]
    roll_points_against_10: Optional[float]
    roll_margin_10: Optional[float]
    roll_win_rate_10: Optional[float]

    roll_points_for_20: Optional[float]
    roll_points_against_20: Optional[float]
    roll_margin_20: Optional[float]
    roll_win_rate_20: Optional[float]

    # Season-to-date aggregates
    season_points_for_avg: Optional[float]
    season_points_against_avg: Optional[float]
    season_margin_avg: Optional[float]

    # Home/away splits
    home_points_for_avg: Optional[float]
    home_points_against_avg: Optional[float]

    # Scheduling features
    rest_days: Optional[int]
    is_b2b: Optional[int]

    # Opponent-adjusted rolling stats
    opp_roll_margin_5: Optional[float]
    opp_roll_margin_10: Optional[float]

    # Strength of schedule
    sos: Optional[float]

    # ELO
    elo: Optional[float]
    opp_elo: Optional[float]

    # Team form
    form_last3: Optional[float]

    # -----------------------------
    # Validators
    # -----------------------------

    @validator("game_id", "team", "opponent")
    def validate_non_empty(cls, v):
        if v is None or str(v).strip() == "":
            raise ValueError("Identifier fields must be non-empty")
        return v

    @validator("season")
    def validate_season_format(cls, v):
        # Expect "2024-25"
        if not isinstance(v, str) or not v.count("-") == 1:
            raise ValueError(f"Invalid season format: {v}")
        return v

    @validator("is_home", "is_b2b")
    def validate_binary(cls, v):
        if v is not None and v not in (0, 1):
            raise ValueError(f"Expected binary 0/1, got {v}")
        return v

    @validator(
        "roll_win_rate_5",
        "roll_win_rate_10",
        "roll_win_rate_20",
    )
    def validate_win_rate(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"Win rate must be in [0,1], got {v}")
        return v

    @validator("rest_days")
    def validate_rest_days(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"rest_days cannot be negative, got {v}")
        return v

    @validator("elo", "opp_elo")
    def validate_elo(cls, v):
        if v is not None and not (500 <= v <= 3000):
            # ELO rarely leaves this range; warn but allow
            raise ValueError(f"ELO value out of expected range: {v}")
        return v
