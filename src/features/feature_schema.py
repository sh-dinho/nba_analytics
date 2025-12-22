# ============================================================
# 🏀 NBA Analytics v3
# Module: Feature Schema
# File: src/features/feature_schema.py
# Author: Sadiq
#
# Description:
#     Defines the strict schema for engineered team-level
#     features to guarantee point-in-time correctness and
#     prevent leakage and malformed features.
# ============================================================

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, validator


class FeatureRow(BaseModel):
    game_id: str
    team: str
    opponent: str
    date: date
    is_home: bool
    rolling_win_rate: float
    rolling_points_for: float
    rolling_points_against: float

    @validator("rolling_win_rate")
    def validate_win_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"rolling_win_rate must be in [0,1], got {v}")
        return v
