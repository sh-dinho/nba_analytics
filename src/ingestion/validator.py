# ============================================================
# 🏀 NBA Analytics v3
# Module: Ingestion Validation
# File: src/ingestion/validator.py
# Author: Sadiq
#
# Description:
#     Strict schema validation and integrity checks for
#     ScoreboardV3 ingestion. Ensures no malformed data enters
#     the canonical snapshots.
# ============================================================

from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, validator


GameStatus = Literal["final", "scheduled", "in_progress"]


class GameSchema(BaseModel):
    game_id: str
    date: date
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    status: GameStatus
    season: str | None = None

    @validator("home_team", "away_team")
    def validate_team(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Team name cannot be empty")
        return v

    @validator("home_score", "away_score")
    def validate_score(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 0:
            raise ValueError("Score cannot be negative")
        return v


def validate_ingestion_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates a raw ingestion dataframe against GameSchema and
    enforces basic integrity constraints (uniqueness, non-null keys).
    """
    required_cols = [
        "game_id",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "status",
        "season",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Ingestion dataframe missing required columns: {missing}")

    # Basic integrity checks
    if not df["game_id"].is_unique:
        dupes = df["game_id"][df["game_id"].duplicated()].unique()
        raise ValueError(f"Duplicate game_id detected in ingestion: {dupes[:5]}")

    if df["home_team"].isna().any():
        raise ValueError("Missing home_team values in ingestion dataframe.")

    if df["away_team"].isna().any():
        raise ValueError("Missing away_team values in ingestion dataframe.")

    # Pydantic validation row by row
    records = df.to_dict(orient="records")
    validated_records: list[dict] = []

    for rec in records:
        model = GameSchema(**rec)
        validated_records.append(model.dict())

    return pd.DataFrame(validated_records)
