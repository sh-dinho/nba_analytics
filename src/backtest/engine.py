# ============================================================
# Project: NBA Analytics & Betting Engine
# Module: Backtesting Engine
# Author: Sadiq
#
# Description:
#     Historical backtesting engine for evaluating model-driven
#     betting strategies using predictions, odds, and outcomes.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config.paths import PREDICTIONS_DIR, ODDS_DIR, LONG_SNAPSHOT


@dataclass
class BacktestConfig:
    starting_bankroll: float = 1000.0
    min_edge: float = 0.03
    kelly_fraction: float = 0.25
    max_stake_fraction: float = 0.05
    market: str = "moneyline"


def american_to_implied(price: float) -> float:
    if price > 0:
        return 100 / (price + 100)
    else:
        return -price / (-price + 100)


def kelly_fraction(win_prob: float, price: float) -> float:
    if price > 0:
        decimal_odds = 1 + price / 100
    else:
        decimal_odds = 1 + 100 / -price

    b = decimal_odds - 1
    p = win_prob
    q = 1 - p

    k = (b * p - q) / b
    return max(0.0, k)


def current_season_date_range() -> tuple[str, str]:
    today = datetime.today().date()
    year = today.year

    if today.month >= 10:
        start = date(year, 10, 1)
    else:
        start = date(year - 1, 10, 1)

    end = today
    return start.isoformat(), end.isoformat()


class Backtester:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> dict:
        df = self._load_joined_data(start_date, end_date)

        if df.empty:
            logger.warning("No data available for backtest in given date range.")
            return {}

        df = self._compute_edges(df)
        bets = self._apply_strategy(df)
        results = self._simulate_bankroll(bets)

        return results

    def _load_joined_data(
        self, start_date: Optional[str], end_date: Optional[str]
    ) -> pd.DataFrame:
        long_df = pd.read_parquet(LONG_SNAPSHOT)
        long_df["date"] = pd.to_datetime(long_df["date"]).dt.date
        long_df = long_df[["game_id", "team", "won", "date"]]

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

        odds_files = sorted(ODDS_DIR.glob("odds_*.parquet"))
        odds_list = []

        for path in odds_files:
            dt_str = path.stem.replace("odds_", "")
            dt = pd.to_datetime(dt_str).date()

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

        joined = odds.merge(
            preds,
            on=["game_id", "team", "date"],
            how="inner",
            suffixes=("_odds", "_pred"),
        )

        joined = joined.merge(
            long_df,
            on=["game_id", "team"],
            how="left",
        )

        joined = joined.dropna(subset=["won"])

        return joined

    def _compute_edges(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["implied_prob"] = df["price"].apply(american_to_implied)
        df["edge"] = df["win_probability"] - df["implied_prob"]
        return df

    def _apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        bankroll = self.config.starting_bankroll

        stakes = []

        for _, row in df.iterrows():
            edge = row["edge"]

            if edge <= self.config.min_edge:
                stakes.append(0.0)
                continue

            k_full = kelly_fraction(row["win_probability"], row["price"])
            k_used = k_full * self.config.kelly_fraction

            stake = bankroll * k_used
            stake = min(stake, bankroll * self.config.max_stake_fraction)

            stakes.append(stake)

        df["stake"] = stakes
        df = df[df["stake"] > 0].copy()

        return df

    def _simulate_bankroll(self, bets: pd.DataFrame) -> dict:
        if bets.empty:
            logger.warning("No bets placed under the current strategy.")
            return {}

        bets = bets.sort_values("date").reset_index(drop=True)

        bankroll = self.config.starting_bankroll
        bankrolls = []
        profits = []
        daily_records = []

        for _, row in bets.iterrows():
            stake = row["stake"]
            price = row["price"]
            won = row["won"]

            if stake <= 0 or bankroll <= 0:
                bankrolls.append(bankroll)
                profits.append(0.0)
                continue

            if price > 0:
                profit = stake * (price / 100) if won == 1 else -stake
            else:
                profit = stake * (100 / -price) if won == 1 else -stake

            bankroll += profit
            bankrolls.append(bankroll)
            profits.append(profit)

            daily_records.append(
                {
                    "date": row["date"],
                    "game_id": row["game_id"],
                    "team": row["team"],
                    "price": price,
                    "win_probability": row["win_probability"],
                    "implied_prob": row["implied_prob"],
                    "edge": row["edge"],
                    "stake": stake,
                    "profit": profit,
                    "bankroll_after": bankroll,
                }
            )

        bets["profit"] = profits
        bets["bankroll_after"] = bankrolls

        total_profit = bankroll - self.config.starting_bankroll
        roi = total_profit / self.config.starting_bankroll
        hit_rate = (bets["profit"] > 0).mean()
        max_drawdown = self._max_drawdown([r["bankroll_after"] for r in daily_records])

        num_bets = len(bets)
        num_wins = int((bets["profit"] > 0).sum())
        num_losses = int((bets["profit"] < 0).sum())
        num_pushes = int((bets["profit"] == 0).sum())

        return {
            "config": self.config,
            "bets": bets,
            "records": pd.DataFrame(daily_records),
            "final_bankroll": bankroll,
            "total_profit": total_profit,
            "roi": roi,
            "hit_rate": hit_rate,
            "max_drawdown": max_drawdown,
            "num_bets": num_bets,
            "num_wins": num_wins,
            "num_losses": num_losses,
            "num_pushes": num_pushes,
        }

    @staticmethod
    def _max_drawdown(series) -> float:
        series = np.array(series, dtype=float)
        if series.size == 0:
            return 0.0
        running_max = np.maximum.accumulate(series)
        drawdowns = (series - running_max) / running_max
        return float(drawdowns.min())
