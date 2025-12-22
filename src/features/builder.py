# ============================================================
# 🏀 NBA Analytics v3
# Module: Feature Engineering
# File: src/features/builder.py
# Author: Sadiq
#
# Description:
#     Builds strictly point-in-time-correct team-level features
#     from the canonical long-format game data. Includes:
#       - rolling win rate
#       - rolling points for/against
#     with no leakage (uses only past games).
#     Validates feature schema and writes optional metadata.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.config.paths import DATA_DIR, LONG_SNAPSHOT
from src.features.feature_schema import FeatureRow

FEATURES_DIR = DATA_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureConfig:
    window_games: int = 10
    version: str = "v1"


class FeatureBuilder:
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def build_from_long(self, df_long: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Builds team-centric rolling features from the canonical long-format dataframe.
        Expects columns: ['game_id', 'date', 'team', 'opponent', 'points_for', 'points_against', 'won', 'is_home'].
        """
        if df_long is None:
            if not LONG_SNAPSHOT.exists():
                raise FileNotFoundError(
                    f"Long-format snapshot not found: {LONG_SNAPSHOT}"
                )
            df_long = pd.read_parquet(LONG_SNAPSHOT)

        required_cols = [
            "game_id",
            "date",
            "team",
            "opponent",
            "points_for",
            "points_against",
            "won",
            "is_home",
        ]
        missing = [c for c in required_cols if c not in df_long.columns]
        if missing:
            raise ValueError(f"Long dataframe missing required columns: {missing}")

        df = df_long.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date

        df = df.sort_values(["team", "date"]).reset_index(drop=True)

        window = self.config.window_games

        df["rolling_win_rate"] = (
            df.groupby("team")["won"]
            .rolling(window=window, min_periods=1, closed="left")
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling_points_for"] = (
            df.groupby("team")["points_for"]
            .rolling(window=window, min_periods=1, closed="left")
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling_points_against"] = (
            df.groupby("team")["points_against"]
            .rolling(window=window, min_periods=1, closed="left")
            .mean()
            .reset_index(level=0, drop=True)
        )

        df_features = df[
            [
                "game_id",
                "team",
                "opponent",
                "date",
                "is_home",
                "rolling_win_rate",
                "rolling_points_for",
                "rolling_points_against",
            ]
        ].copy()

        self._validate_features(df_features)
        self._write_latest_snapshot(df_features)

        return df_features

    def _validate_features(self, df: pd.DataFrame):
        """
        Validate each row against FeatureRow schema. This is strict and may be
        expensive on huge datasets, but it's safe for daily snapshots and
        consulting-grade correctness.
        """
        required_cols = [
            "game_id",
            "team",
            "opponent",
            "date",
            "is_home",
            "rolling_win_rate",
            "rolling_points_for",
            "rolling_points_against",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature dataframe missing required columns: {missing}")

        records = df.to_dict(orient="records")
        for rec in records:
            FeatureRow(**rec)

        logger.info(f"Feature validation passed for {len(df)} rows.")

    def _write_latest_snapshot(self, df: pd.DataFrame):
        """
        Writes the latest feature snapshot and metadata (version, generated_at).
        """
        version_dir = FEATURES_DIR / self.config.version
        version_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        snapshot_path = version_dir / f"features_{ts}.parquet"
        latest_path = version_dir / "features_latest.parquet"
        meta_path = version_dir / "features_metadata.json"

        df.to_parquet(snapshot_path, index=False)
        df.to_parquet(latest_path, index=False)

        metadata = {
            "version": self.config.version,
            "generated_at_utc": datetime.utcnow().isoformat(),
            "rows": int(len(df)),
            "columns": df.columns.tolist(),
            "snapshot_path": str(snapshot_path),
        }

        import json

        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logger.success(f"Feature snapshot written → {snapshot_path}")
        logger.info(f"Feature latest alias updated → {latest_path}")
        logger.info(f"Feature metadata written → {meta_path}")
