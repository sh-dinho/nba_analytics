from __future__ import annotations
# ============================================================
# 🏀 NBA Analytics v3
# Module: Backtesting — Spread (ATS)
# File: src/backtest/engine_spread.py
# Author: Sadiq
#
# Description:
#     Backtests ATS predictions using:
#       - predicted_margin
#       - market_spread
#       - actual margin
#
#     Computes:
#       - ROI
#       - hit rate
#       - max drawdown
# ============================================================

from dataclasses import dataclass
from src.config.paths import (
    DATA_DIR,
)

SPREAD_PRED_DIR = DATA_DIR / "predictions_spread"


@dataclass
class SpreadBacktestConfig:
    starting_bankroll: float = 1000.0
    min_edge: float = 2.0  # points
    stake_fraction: float = 0.02


class SpreadBacktester:
    def __init__(self, cfg: SpreadBacktestConfig):
        self.cfg = cfg

    def run(self, start_date=None, end_date=None):
        logger.info(f"🏀 Running spread backtest start={start_date}, end={end_date}")

        df = self._load_joined_data(start_date, end_date)
        if df.empty:
            return {}

        df = self._compute_edges(df)
        bets = self._apply_strategy(df)
        return self._simulate_bankroll(bets)

    def _load_joined_data(self, start_date, end_date):
        pred_files = sorted(SPREAD_PRED_DIR.glob("spread_*.parquet"))
        preds_list = []

        for path in pred_files:
            dt = path.stem.replace("spread_", "")
            dt = pd.to_datetime(dt).date()

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

        odds_files = sorted(ODDS_DIR.glob("odds_*.parquet"))
        odds_list = []

        for path in odds_files:
            dt = path.stem.replace("odds_", "")
            dt = pd.to_datetime(dt).date()

            if start_date and dt < pd.to_datetime(start_date).date():
                continue
            if end_date and dt > pd.to_datetime(end_date).date():
                continue

            df = pd.read_parquet(path)
            df["date"] = dt
            odds_list.append(df)

        if not odds_list:
            return pd.DataFrame()

        odds = pd.concat(odds_list, ignore_index=True)

        long_df = pd.read_parquet(LONG_SNAPSHOT)
        long_df["date"] = pd.to_datetime(long_df["date"]).dt.date

        home_rows = long_df[long_df["is_home"] == True].copy()
        home_rows["actual_margin"] = (
            home_rows["points_for"] - home_rows["points_against"]
        )

        joined = preds.merge(
            odds[["game_id", "market_spread"]],
            on="game_id",
            how="inner",
        ).merge(
            home_rows[["game_id", "date", "actual_margin"]],
            on=["game_id", "date"],
            how="inner",
        )

        return joined

    def _compute_edges(self, df):
        df = df.copy()
        df["edge_home"] = df["predicted_margin"] - df["market_spread"]
        df["edge_away"] = df["market_spread"] - df["predicted_margin"]
        df["edge"] = df[["edge_home", "edge_away"]].max(axis=1)
        df["direction"] = df.apply(
            lambda r: "HOME" if r["edge_home"] > r["edge_away"] else "AWAY",
            axis=1,
        )
        return df

    def _apply_strategy(self, df):
        df = df[df["edge"] >= self.cfg.min_edge].copy()
        df["stake"] = self.cfg.starting_bankroll * self.cfg.stake_fraction
        return df

    def _simulate_bankroll(self, bets):
        if bets.empty:
            return {}

        bankroll = self.cfg.starting_bankroll
        bankrolls = []
        profits = []

        for _, row in bets.iterrows():
            stake = row["stake"]
            actual = row["actual_margin"]
            spread = row


# ============================================================
# 🏀 NBA Analytics v3
# Module: Backtesting — Spread (ATS)
# File: src/backtest/engine_spread.py
# Author: Sadiq
#
# Description:
#     Backtests ATS predictions using:
#       - predicted_margin
#       - market_spread
#       - actual margin
#
#     Computes:
#       - ROI
#       - hit rate
#       - max drawdown
# ============================================================


from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config.paths import (
    LONG_SNAPSHOT,
    DATA_DIR,
    ODDS_DIR,
)

SPREAD_PRED_DIR = DATA_DIR / "predictions_spread"


@dataclass
class SpreadBacktestConfig:
    starting_bankroll: float = 1000.0
    min_edge: float = 2.0  # points
    stake_fraction: float = 0.02


class SpreadBacktester:
    def __init__(self, cfg: SpreadBacktestConfig):
        self.cfg = cfg

    def run(self, start_date=None, end_date=None):
        logger.info(f"🏀 Running spread backtest start={start_date}, end={end_date}")

        df = self._load_joined_data(start_date, end_date)
        if df.empty:
            return {}

        df = self._compute_edges(df)
        bets = self._apply_strategy(df)
        return self._simulate_bankroll(bets)

    def _load_joined_data(self, start_date, end_date):
        pred_files = sorted(SPREAD_PRED_DIR.glob("spread_*.parquet"))
        preds_list = []

        for path in pred_files:
            dt = path.stem.replace("spread_", "")
            dt = pd.to_datetime(dt).date()

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

        odds_files = sorted(ODDS_DIR.glob("odds_*.parquet"))
        odds_list = []

        for path in odds_files:
            dt = path.stem.replace("odds_", "")
            dt = pd.to_datetime(dt).date()

            if start_date and dt < pd.to_datetime(start_date).date():
                continue
            if end_date and dt > pd.to_datetime(end_date).date():
                continue

            df = pd.read_parquet(path)
            df["date"] = dt
            odds_list.append(df)

        if not odds_list:
            return pd.DataFrame()

        odds = pd.concat(odds_list, ignore_index=True)

        long_df = pd.read_parquet(LONG_SNAPSHOT)
        long_df["date"] = pd.to_datetime(long_df["date"]).dt.date

        home_rows = long_df[long_df["is_home"] == True].copy()
        home_rows["actual_margin"] = (
            home_rows["points_for"] - home_rows["points_against"]
        )

        joined = preds.merge(
            odds[["game_id", "market_spread"]],
            on="game_id",
            how="inner",
        ).merge(
            home_rows[["game_id", "date", "actual_margin"]],
            on=["game_id", "date"],
            how="inner",
        )

        return joined

    def _compute_edges(self, df):
        df = df.copy()
        df["edge_home"] = df["predicted_margin"] - df["market_spread"]
        df["edge_away"] = df["market_spread"] - df["predicted_margin"]
        df["edge"] = df[["edge_home", "edge_away"]].max(axis=1)
        df["direction"] = df.apply(
            lambda r: "HOME" if r["edge_home"] > r["edge_away"] else "AWAY",
            axis=1,
        )
        return df

    def _apply_strategy(self, df):
        df = df[df["edge"] >= self.cfg.min_edge].copy()
        df["stake"] = self.cfg.starting_bankroll * self.cfg.stake_fraction
        return df

    def _simulate_bankroll(self, bets):
        if bets.empty:
            return {}

        bankroll = self.cfg.starting_bankroll
        bankrolls = []
        profits = []

        for _, row in bets.iterrows():
            stake = row["stake"]
            actual = row["actual_margin"]
            spread = row
