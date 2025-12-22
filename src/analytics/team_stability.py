# ============================================================
# 🏀 NBA Analytics v3
# Module: Team Stability Engine
# File: src/analytics/team_stability.py
# Author: Sadiq
#
# Description:
#     Computes team-level stability metrics using backtest
#     records and predictions:
#       - ROI per team
#       - Profit per team
#       - Bet count
#       - Volatility of edge and win_probability
#       - Hit rate
#       - Stability score (0–100)
#
#     Produces:
#       - teams_to_avoid
#       - teams_to_watch
#       - full per-team metrics table
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.engine import Backtester, BacktestConfig


@dataclass
class TeamStabilityConfig:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    min_bets: int = 20
    avoid_roi_threshold: float = -0.05
    watch_roi_threshold: float = 0.05
    volatility_weight: float = 0.5
    roi_weight: float = 0.5


@dataclass
class TeamStabilityResult:
    teams: pd.DataFrame
    teams_to_avoid: pd.DataFrame
    teams_to_watch: pd.DataFrame


class TeamStabilityEngine:
    def __init__(self, cfg: Optional[TeamStabilityConfig] = None):
        self.cfg = cfg or TeamStabilityConfig()

    def run(self) -> TeamStabilityResult:
        logger.info(
            f"🏀 Computing team stability with config: "
            f"start={self.cfg.start_date}, end={self.cfg.end_date}, "
            f"min_bets={self.cfg.min_bets}"
        )

        bt_cfg = BacktestConfig(
            starting_bankroll=1000.0,
            min_edge=0.0,
            kelly_fraction=0.25,
            max_stake_fraction=0.05,
        )
        bt = Backtester(bt_cfg)

        res = bt.run(
            start_date=self.cfg.start_date,
            end_date=self.cfg.end_date,
            model_name=None,
            model_version=None,
        )

        if not res or res.get("records") is None:
            logger.warning(
                "No backtest records available for team stability computation."
            )
            empty = pd.DataFrame()
            return TeamStabilityResult(empty, empty, empty)

        records = res["records"].copy()
        if records.empty:
            logger.warning("Backtest records are empty for team stability computation.")
            empty = pd.DataFrame()
            return TeamStabilityResult(empty, empty, empty)

        teams = self._compute_team_metrics(records)
        teams = self._compute_stability_score(teams)

        teams_filtered = teams[teams["num_bets"] >= self.cfg.min_bets].copy()

        teams_to_avoid = teams_filtered[
            (teams_filtered["roi"] <= self.cfg.avoid_roi_threshold)
        ].sort_values("stability_score", ascending=True)

        teams_to_watch = teams_filtered[
            (teams_filtered["roi"] >= self.cfg.watch_roi_threshold)
        ].sort_values("stability_score", ascending=False)

        return TeamStabilityResult(
            teams=teams,
            teams_to_avoid=teams_to_avoid,
            teams_to_watch=teams_to_watch,
        )

    @staticmethod
    def _compute_team_metrics(records: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates per-team stats from bet-level records.
        Expected columns: team, stake, profit, win_probability, edge, date.
        """
        df = records.copy()

        # Basic aggregations
        grouped = df.groupby("team").agg(
            total_staked=("stake", "sum"),
            total_profit=("profit", "sum"),
            num_bets=("stake", lambda x: (x > 0).sum()),
            mean_edge=("edge", "mean"),
            std_edge=("edge", "std"),
            mean_win_prob=("win_probability", "mean"),
            std_win_prob=("win_probability", "std"),
            wins=("profit", lambda x: (x > 0).sum()),
        )

        grouped["roi"] = grouped["total_profit"] / grouped["total_staked"].replace(
            0, np.nan
        )
        grouped["roi"] = grouped["roi"].fillna(0.0)
        grouped["hit_rate"] = grouped["wins"] / grouped["num_bets"].replace(0, np.nan)
        grouped["hit_rate"] = grouped["hit_rate"].fillna(0.0)

        # Volatility proxy: combine std of edge and win_probability
        grouped["volatility"] = (
            grouped["std_edge"].fillna(0.0).abs()
            + grouped["std_win_prob"].fillna(0.0).abs()
        )

        grouped = grouped.reset_index()

        return grouped

    def _compute_stability_score(self, teams: pd.DataFrame) -> pd.DataFrame:
        """
        Combines ROI and volatility into a single Stability Score [0,100].
        Higher = more stable + profitable.
        """
        df = teams.copy()

        if df.empty:
            df["stability_score"] = 0.0
            return df

        # Normalize ROI to [0,1] based on percentiles
        roi = df["roi"].fillna(0.0)
        roi_low = roi.quantile(0.05)
        roi_high = roi.quantile(0.95)
        roi_norm = (roi - roi_low) / (roi_high - roi_low + 1e-9)
        roi_norm = roi_norm.clip(0.0, 1.0)

        # Normalize volatility (lower volatility is better)
        vol = df["volatility"].fillna(0.0)
        vol_low = vol.quantile(0.05)
        vol_high = vol.quantile(0.95)
        vol_norm = (vol - vol_low) / (vol_high - vol_low + 1e-9)
        vol_norm = vol_norm.clip(0.0, 1.0)
        stability_component = 1.0 - vol_norm

        score = (
            self.cfg.roi_weight * roi_norm
            + self.cfg.volatility_weight * stability_component
        )

        df["stability_score"] = (score * 100).round(1)

        return df
