from __future__ import annotations

# ============================================================
# 🏀 NBA Analytics
# Module: Accuracy Engine
# File: src/backtest/accuracy.py
# Author: Sadiq
#
# Description:
#     Computes classification accuracy and related metrics for
#     model predictions vs actual outcomes.
#
#     Supports:
#       • overall accuracy
#       • per-season accuracy
#       • per-team accuracy
#       • per-model accuracy
#       • Brier score
#       • log loss
#       • calibration buckets
#
#     Inputs:
#       • predictions (PREDICTIONS_DIR)
#       • canonical results snapshot
#
#     Outputs:
#       • AccuracyResult dataclass
#       • raw merged dataframe for downstream reporting
# ============================================================

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
from loguru import logger

from src.config.paths import (
    PREDICTIONS_DIR,
    RESULTS_SNAPSHOT_DIR,
)


@dataclass
class AccuracyResult:
    overall_accuracy: float
    total_examples: int
    brier_score: float
    log_loss: float
    by_season: pd.DataFrame
    by_team: pd.DataFrame
    by_model: pd.DataFrame
    calibration: pd.DataFrame
    raw: pd.DataFrame


class AccuracyEngine:
    def __init__(self, threshold: float = 0.5, n_calibration_bins: int = 10):
        self.threshold = threshold
        self.n_calibration_bins = n_calibration_bins

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> AccuracyResult:

        logger.info(
            f"🏀 Running accuracy engine: start={start_date}, "
            f"end={end_date}, model_version={model_version}, "
            f"threshold={self.threshold}"
        )

        preds = self._load_predictions(start_date, end_date, model_version)
        results = self._load_results(start_date, end_date)

        if preds.empty or results.empty:
            logger.warning("AccuracyEngine: no data available for evaluation.")
            return self._empty_result()

        merged = self._merge_predictions_and_results(preds, results)

        if merged.empty:
            logger.warning("AccuracyEngine: merged dataset is empty.")
            return self._empty_result()

        merged = self._compute_core_fields(merged)

        overall_acc = float(merged["correct"].mean())
        total_examples = int(len(merged))
        brier = float(((merged["win_probability"] - merged["won"]) ** 2).mean())
        log_loss = self._compute_log_loss(merged["won"], merged["win_probability"])

        by_season = self._agg_by(merged, "season")
        by_team = self._agg_by(merged, "team")
        by_model = self._agg_by(merged, "model_version")
        calibration = self._compute_calibration(merged)

        return AccuracyResult(
            overall_accuracy=overall_acc,
            total_examples=total_examples,
            brier_score=brier,
            log_loss=log_loss,
            by_season=by_season,
            by_team=by_team,
            by_model=by_model,
            calibration=calibration,
            raw=merged,
        )

    # --------------------------------------------------------
    # Data loading
    # --------------------------------------------------------
    @staticmethod
    def _load_results(start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        try:
            df = pd.read_parquet(RESULTS_SNAPSHOT_DIR / "results.parquet")
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df
        except Exception as e:
            logger.error(f"Failed to load results snapshot: {e}")
            return pd.DataFrame()

    @staticmethod
    def _load_predictions(
        start_date: Optional[str],
        end_date: Optional[str],
        model_version: Optional[str],
    ) -> pd.DataFrame:

        pred_files = sorted(PREDICTIONS_DIR.glob("predictions_*.parquet"))
        preds_list: List[pd.DataFrame] = []

        start = pd.to_datetime(start_date).date() if start_date else None
        end = pd.to_datetime(end_date).date() if end_date else None

        for path in pred_files:
            try:
                dt_str = path.stem.replace("predictions_", "")
                dt = pd.to_datetime(dt_str).date()
            except Exception:
                continue

            if start and dt < start:
                continue
            if end and dt > end:
                continue

            df = pd.read_parquet(path)
            df["date"] = dt

            if model_version is not None:
                df = df[df["model_version"] == model_version]

            if not df.empty:
                preds_list.append(df)

        if not preds_list:
            return pd.DataFrame()

        preds = pd.concat(preds_list, ignore_index=True)
        logger.info(f"Loaded {len(preds)} prediction rows for accuracy evaluation.")
        return preds

    # --------------------------------------------------------
    # Merge predictions + results
    # --------------------------------------------------------
    @staticmethod
    def _merge_predictions_and_results(
        preds: pd.DataFrame,
        results: pd.DataFrame,
    ) -> pd.DataFrame:

        required_pred_cols = {"game_id", "team", "date", "win_probability"}
        required_res_cols = {"game_id", "team", "date", "season", "won"}

        missing_pred = required_pred_cols - set(preds.columns)
        missing_res = required_res_cols - set(results.columns)

        if missing_pred:
            raise ValueError(f"Predictions missing required columns: {missing_pred}")
        if missing_res:
            raise ValueError(f"Results snapshot missing required columns: {missing_res}")

        merged = preds.merge(
            results[list(required_res_cols)],
            on=["game_id", "team", "date"],
            how="inner",
        )

        return merged

    def _compute_core_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["predicted_label"] = (df["win_probability"] >= self.threshold).astype(int)
        df["correct"] = (df["predicted_label"] == df["won"]).astype(int)
        return df

    # --------------------------------------------------------
    # Aggregations
    # --------------------------------------------------------
    @staticmethod
    def _agg_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns:
            return pd.DataFrame(columns=[col, "accuracy", "count"])

        grouped = df.groupby(col)
        out = pd.DataFrame({
            col: grouped.size().index,
            "accuracy": grouped["correct"].mean().values,
            "count": grouped.size().values,
        })

        return out.sort_values("accuracy", ascending=False).reset_index(drop=True)

    # --------------------------------------------------------
    # Calibration + log loss
    # --------------------------------------------------------
    def _compute_calibration(self, df: pd.DataFrame) -> pd.DataFrame:
        probs = df["win_probability"].clip(0.0, 1.0)
        labels = df["won"]

        bins = np.linspace(0.0, 1.0, self.n_calibration_bins + 1)
        bin_ids = np.digitize(probs, bins) - 1

        rows = []
        for b in range(self.n_calibration_bins):
            mask = bin_ids == b
            if not mask.any():
                continue

            rows.append({
                "bin_index": b,
                "bin_start": float(bins[b]),
                "bin_end": float(bins[b + 1]),
                "mean_predicted_prob": float(probs[mask].mean()),
                "empirical_accuracy": float(labels[mask].mean()),
                "count": int(mask.sum()),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _compute_log_loss(y_true: pd.Series, y_prob: pd.Series) -> float:
        eps = 1e-15
        p = y_prob.clip(eps, 1 - eps)
        y = y_true.astype(float)
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(loss.mean())

    # --------------------------------------------------------
    # Empty result
    # --------------------------------------------------------
    @staticmethod
    def _empty_result() -> AccuracyResult:
        empty = pd.DataFrame()
        return AccuracyResult(
            overall_accuracy=0.0,
            total_examples=0,
            brier_score=0.0,
            log_loss=0.0,
            by_season=empty,
            by_team=empty,
            by_model=empty,
            calibration=empty,
            raw=empty,
        )