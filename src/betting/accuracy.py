from __future__ import annotations
# ============================================================
# 🏀 NBA Analytics v3
# Module: Accuracy Engine
# File: src/backtest/accuracy.py
# Author: Sadiq
#
# Description:
#     Computes classification accuracy for model predictions
#     vs actual outcomes. Supports:
#       - overall accuracy
#       - per-season accuracy
#       - sample predictions table
# ============================================================


from dataclasses import dataclass
from datetime import date
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

    def run(self, start_date: Optional[str], end_date: Optional[str]) -> AccuracyResult:
        long_df = pd.read_parquet(LONG_SNAPSHOT)
        long_df["date"] = pd.to_datetime(long_df["date"]).dt.date

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
            return AccuracyResult(0.0, 0, pd.DataFrame(), pd.DataFrame())

        preds = pd.concat(preds_list, ignore_index=True)

        merged = preds.merge(
            long_df[["game_id", "team", "won", "date", "season"]],
            on=["game_id", "team", "date"],
            how="inner",
        )

        if merged.empty:
            return AccuracyResult(0.0, 0, pd.DataFrame(), pd.DataFrame())

        merged["predicted_win"] = (merged["win_probability"] >= self.threshold).astype(
            int
        )
        merged["correct"] = (merged["predicted_win"] == merged["won"]).astype(int)

        overall = merged["correct"].mean()
        total = len(merged)

        by_season = (
            merged.groupby("season")["correct"]
            .mean()
            .reset_index()
            .rename(columns={"correct": "accuracy"})
        )

        return AccuracyResult(
            overall_accuracy=float(overall),
            total_examples=total,
            by_season=by_season,
            raw=merged,
        )
