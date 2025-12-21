# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Build aggregated, dashboard-ready data for
#              bankroll curves, ROI, volume, and calibration.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.config.paths import LOGS_DIR, PREDICTIONS_DIR


# ------------------------------------------------------------
# Data containers
# ------------------------------------------------------------


@dataclass
class TimeSeriesPoint:
    date: str
    value: float


@dataclass
class DashboardBundle:
    bankroll_curve: List[TimeSeriesPoint]
    daily_roi: List[TimeSeriesPoint]
    bet_volume: List[TimeSeriesPoint]
    hit_rate: List[TimeSeriesPoint]
    exposure: List[TimeSeriesPoint]
    prediction_distribution: Dict[str, Any]
    calibration_curve: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bankroll_curve": [asdict(p) for p in self.bankroll_curve],
            "daily_roi": [asdict(p) for p in self.daily_roi],
            "bet_volume": [asdict(p) for p in self.bet_volume],
            "hit_rate": [asdict(p) for p in self.hit_rate],
            "exposure": [asdict(p) for p in self.exposure],
            "prediction_distribution": self.prediction_distribution,
            "calibration_curve": self.calibration_curve,
        }


# ------------------------------------------------------------
# Dashboard data builder
# ------------------------------------------------------------


class DashboardDataBuilder:
    """
    Aggregates logs and predictions into dashboard-ready structures:

      - Bankroll curve
      - Daily ROI
      - Bet volume over time
      - Hit rate over time
      - Exposure over time
      - Prediction histograms
      - Calibration curves
    """

    def __init__(self):
        self.bet_log_path = LOGS_DIR / "bets.parquet"

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def build(self) -> DashboardBundle:
        bets = self._load_bets()
        preds = self._load_predictions_for_calibration()

        bankroll_curve = self._build_bankroll_curve(bets)
        daily_roi = self._build_daily_roi(bets)
        bet_volume = self._build_bet_volume(bets)
        hit_rate = self._build_hit_rate(bets)
        exposure = self._build_exposure(bets)
        prediction_distribution = self._build_prediction_distribution(preds)
        calibration_curve = self._build_calibration_curve(preds)

        bundle = DashboardBundle(
            bankroll_curve=bankroll_curve,
            daily_roi=daily_roi,
            bet_volume=bet_volume,
            hit_rate=hit_rate,
            exposure=exposure,
            prediction_distribution=prediction_distribution,
            calibration_curve=calibration_curve,
        )
        logger.info("DashboardDataBuilder: dashboard bundle constructed.")
        return bundle

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------
    def _load_bets(self) -> pd.DataFrame:
        if not self.bet_log_path.exists():
            logger.info(
                "DashboardDataBuilder: bet log not found; returning empty DataFrame."
            )
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "prediction_date",
                    "stake",
                    "profit",
                    "won",
                ]
            )

        df = pd.read_parquet(self.bet_log_path)
        if df.empty:
            logger.info("DashboardDataBuilder: bet log is empty.")
            return df

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        # Use prediction_date as primary x-axis for most aggregates
        df["prediction_date"] = pd.to_datetime(
            df["prediction_date"], errors="coerce"
        ).dt.date

        return df

    def _load_predictions_for_calibration(self) -> pd.DataFrame:
        files = sorted(PREDICTIONS_DIR.glob("predictions_*_v*.parquet"))
        if not files:
            logger.info(
                "DashboardDataBuilder: no prediction files found for calibration."
            )
            return pd.DataFrame(columns=["date", "win_probability", "won"])

        # For calibration we can concatenate a subset (e.g., last N files)
        # Here: last 30 prediction files
        files = files[-30:]
        frames = []
        for f in files:
            try:
                df = pd.read_parquet(f)
                frames.append(df[["date", "win_probability", "won"]])
            except Exception as e:
                logger.warning(f"DashboardDataBuilder: failed to read {f.name}: {e}")

        if not frames:
            return pd.DataFrame(columns=["date", "win_probability", "won"])

        preds = pd.concat(frames, ignore_index=True)
        preds["date"] = pd.to_datetime(preds["date"], errors="coerce").dt.date
        return preds

    # --------------------------------------------------------
    # Bankroll curve
    # --------------------------------------------------------
    def _build_bankroll_curve(self, bets: pd.DataFrame) -> List[TimeSeriesPoint]:
        if bets.empty:
            return []

        # Use prediction_date as the day the bet belongs to
        df = bets.copy()
        df = df.dropna(subset=["prediction_date"])
        if df.empty:
            return []

        # Only resolved bets (profit not null)
        df = df.dropna(subset=["profit"])
        if df.empty:
            return []

        # Sort by prediction_date, then timestamp
        df = df.sort_values(["prediction_date", "timestamp"])

        # Cumulative profit over time
        df["cumulative_profit"] = df["profit"].cumsum()

        series = df.groupby("prediction_date", as_index=False)[
            "cumulative_profit"
        ].last()

        return [
            TimeSeriesPoint(
                date=str(row["prediction_date"]), value=float(row["cumulative_profit"])
            )
            for _, row in series.iterrows()
        ]

    # --------------------------------------------------------
    # Daily ROI
    # --------------------------------------------------------
    def _build_daily_roi(self, bets: pd.DataFrame) -> List[TimeSeriesPoint]:
        if bets.empty:
            return []

        df = bets.copy()
        df = df.dropna(subset=["prediction_date"])
        if df.empty:
            return []

        df_resolved = df.dropna(subset=["profit"])
        if df_resolved.empty:
            return []

        grouped = df_resolved.groupby("prediction_date", as_index=False).agg(
            total_stake=("stake", "sum"),
            total_profit=("profit", "sum"),
        )

        grouped["roi"] = grouped.apply(
            lambda r: (
                r["total_profit"] / r["total_stake"] if r["total_stake"] > 0 else np.nan
            ),
            axis=1,
        )

        series = grouped.dropna(subset=["roi"])

        return [
            TimeSeriesPoint(date=str(row["prediction_date"]), value=float(row["roi"]))
            for _, row in series.iterrows()
        ]

    # --------------------------------------------------------
    # Bet volume
    # --------------------------------------------------------
    def _build_bet_volume(self, bets: pd.DataFrame) -> List[TimeSeriesPoint]:
        if bets.empty:
            return []

        df = bets.copy()
        df = df.dropna(subset=["prediction_date"])
        if df.empty:
            return []

        grouped = df.groupby("prediction_date", as_index=False).agg(
            count=("game_id", "count"),
        )

        return [
            TimeSeriesPoint(date=str(row["prediction_date"]), value=float(row["count"]))
            for _, row in grouped.iterrows()
        ]

    # --------------------------------------------------------
    # Hit rate
    # --------------------------------------------------------
    def _build_hit_rate(self, bets: pd.DataFrame) -> List[TimeSeriesPoint]:
        if bets.empty:
            return []

        df = bets.copy()
        df = df.dropna(subset=["prediction_date"])
        if df.empty:
            return []

        df_resolved = df.dropna(subset=["won"])
        if df_resolved.empty:
            return []

        grouped = df_resolved.groupby("prediction_date", as_index=False).agg(
            hit_rate=("won", "mean"),
        )

        return [
            TimeSeriesPoint(
                date=str(row["prediction_date"]), value=float(row["hit_rate"])
            )
            for _, row in grouped.iterrows()
        ]

    # --------------------------------------------------------
    # Exposure over time
    # --------------------------------------------------------
    def _build_exposure(self, bets: pd.DataFrame) -> List[TimeSeriesPoint]:
        if bets.empty:
            return []

        df = bets.copy()
        df = df.dropna(subset=["prediction_date"])
        if df.empty:
            return []

        grouped = df.groupby("prediction_date", as_index=False).agg(
            exposure=("stake", "sum"),
        )

        return [
            TimeSeriesPoint(
                date=str(row["prediction_date"]), value=float(row["exposure"])
            )
            for _, row in grouped.iterrows()
        ]

    # --------------------------------------------------------
    # Prediction distribution (histogram)
    # --------------------------------------------------------
    def _build_prediction_distribution(self, preds: pd.DataFrame) -> Dict[str, Any]:
        if preds.empty:
            return {"bins": [], "counts": []}

        df = preds.copy()
        df = df.dropna(subset=["win_probability"])
        if df.empty:
            return {"bins": [], "counts": []}

        probs = df["win_probability"].astype(float).clip(0.0, 1.0).values
        counts, bin_edges = np.histogram(probs, bins=10, range=(0.0, 1.0))

        return {
            "bins": bin_edges.tolist(),
            "counts": counts.tolist(),
        }

    # --------------------------------------------------------
    # Calibration curve
    # --------------------------------------------------------
    def _build_calibration_curve(self, preds: pd.DataFrame) -> Dict[str, Any]:
        if preds.empty:
            return {"mean_pred": [], "frac_pos": []}

        df = preds.copy()
        df = df.dropna(subset=["win_probability", "won"])
        if df.empty:
            return {"mean_pred": [], "frac_pos": []}

        df["win_probability"] = df["win_probability"].astype(float).clip(0.0, 1.0)
        df["won"] = df["won"].astype(int)

        # Bin probabilities into deciles
        bins = np.linspace(0.0, 1.0, 11)
        df["bin"] = pd.cut(df["win_probability"], bins=bins, include_lowest=True)

        grouped = df.groupby("bin")
        mean_pred = grouped["win_probability"].mean()
        frac_pos = grouped["won"].mean()

        # Drop empty bins
        valid = (~mean_pred.isna()) & (~frac_pos.isna())
        mean_pred = mean_pred[valid]
        frac_pos = frac_pos[valid]

        return {
            "mean_pred": mean_pred.values.tolist(),
            "frac_pos": frac_pos.values.tolist(),
        }


if __name__ == "__main__":
    builder = DashboardDataBuilder()
    bundle = builder.build()
    print(bundle.to_dict())
