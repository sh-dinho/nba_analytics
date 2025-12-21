# ============================================================
# Project: NBA Analytics & Betting Engine
# Module: Accuracy Metrics
# Author: Sadiq
#
# Description:
#     Computes model accuracy metrics over historical predictions.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger

from src.config.paths import PREDICTIONS_DIR, LONG_SNAPSHOT


@dataclass
class AccuracyResult:
    overall_accuracy: float
    total_examples: int
    by_season: pd.DataFrame
    raw: pd.DataFrame


class AccuracyEngine:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def run(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> AccuracyResult:
        df = self._load_joined_predictions(start_date, end_date)

        if df.empty:
            logger.warning("No data for accuracy computation.")
            return AccuracyResult(0.0, 0, pd.DataFrame(), df)

        df["predicted_win"] = (df["win_probability"] >= self.threshold).astype(int)
        df["correct"] = (df["predicted_win"] == df["won"]).astype(int)

        overall_accuracy = df["correct"].mean()
        total_examples = len(df)

        by_season = (
            df.groupby("season")["correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n"})
            .reset_index()
        )

        return AccuracyResult(
            overall_accuracy=overall_accuracy,
            total_examples=total_examples,
            by_season=by_season,
            raw=df,
        )

    def _load_joined_predictions(
        self, start_date: Optional[str], end_date: Optional[str]
    ) -> pd.DataFrame:
        long_df = pd.read_parquet(LONG_SNAPSHOT)
        long_df["date"] = pd.to_datetime(long_df["date"]).dt.date

        if "season" not in long_df.columns:
            long_df["season"] = long_df["date"].apply(
                lambda d: (
                    f"{d.year}-{d.year+1}" if d.month >= 10 else f"{d.year-1}-{d.year}"
                )
            )

        long_df = long_df[["game_id", "team", "won", "date", "season"]]

        pred_files = sorted(PREDICTIONS_DIR.glob("predictions_*.parquet"))
        preds_list = []

        for path in pred_files:
            dt_str = path.stem.replace("predictions_", "")
            dt = pd.to_datetime(dt_str).date()
            if start_date and dt < pd.to_datetime(start_date).date():
                continue
            if end_date and dt > pd.to_datetime(end_date).date():
                continue

            df = pd.read_parquet(path)
            df["date"] = dt
            preds_list.append(df)

        if not preds_list:
            return pd.DataFrame()

        preds = pd.concat(preds_list, ignore_index=True)

        joined = preds.merge(
            long_df,
            on=["game_id", "team", "date"],
            how="inner",
        )

        return joined
