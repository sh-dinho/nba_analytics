# ============================================================
# Project: NBA Analytics & Betting Engine
# Author: Sadiq
# Description: Append-only logging of executed bets, outcomes,
#              profit, ROI, and daily performance summaries.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
from loguru import logger

from src.config.paths import LOGS_DIR


# ------------------------------------------------------------
# Schema definition
# ------------------------------------------------------------

BET_LOG_COLUMNS = [
    "timestamp",
    "game_id",
    "market_team",
    "market_side",
    "ml",
    "stake",
    "model_prob",
    "implied_prob",
    "edge",
    "ev_per_unit",
    "kelly_fraction",
    "confidence",
    "prediction_date",
    "model_version",
    "feature_version",
    "won",  # outcome (0/1 or None)
    "profit",  # profit/loss
]


@dataclass
class BetLogEntry:
    timestamp: str
    game_id: str
    market_team: str
    market_side: str
    ml: int
    stake: float
    model_prob: float
    implied_prob: float
    edge: float
    ev_per_unit: float
    kelly_fraction: float
    confidence: float
    prediction_date: str
    model_version: str
    feature_version: str
    won: Optional[int] = None
    profit: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------
# Core logger
# ------------------------------------------------------------


class BetLogger:
    """
    Append-only logger for executed bets and outcomes.
    Produces:
      - bets.parquet (full history)
      - daily summaries
      - ROI tracking
    """

    def __init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOGS_DIR / "bets.parquet"

    # --------------------------------------------------------
    # Load / Save
    # --------------------------------------------------------
    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=BET_LOG_COLUMNS)
        return pd.read_parquet(self.path)

    def save(self, df: pd.DataFrame):
        df.to_parquet(self.path, index=False)
        logger.success(f"BetLogger: updated bet log → {self.path}")

    # --------------------------------------------------------
    # Append executed bets
    # --------------------------------------------------------
    def append_executed_bets(self, executed_df: pd.DataFrame):
        """
        executed_df comes from auto_bet.execute_bets()
        """
        if executed_df is None or executed_df.empty:
            logger.info("BetLogger: no executed bets to append.")
            return

        log_df = self.load()

        # Ensure required columns exist
        for col in BET_LOG_COLUMNS:
            if col not in executed_df.columns:
                executed_df[col] = None

        # Append
        combined = pd.concat([log_df, executed_df[BET_LOG_COLUMNS]], ignore_index=True)
        self.save(combined)

        logger.info(f"BetLogger: appended {len(executed_df)} executed bets.")

    # --------------------------------------------------------
    # Update outcomes (after games finish)
    # --------------------------------------------------------
    def update_outcomes(self, results_df: pd.DataFrame):
        """
        results_df must contain:
            - game_id
            - won (0/1)
        """
        if results_df is None or results_df.empty:
            logger.info("BetLogger: no outcomes to update.")
            return

        log_df = self.load()
        if log_df.empty:
            logger.warning("BetLogger: no existing bets to update outcomes for.")
            return

        if "game_id" not in results_df.columns or "won" not in results_df.columns:
            raise ValueError("results_df must contain columns: game_id, won")

        # Merge outcomes
        updated = log_df.merge(
            results_df[["game_id", "won"]],
            on="game_id",
            how="left",
            suffixes=("", "_new"),
        )

        # Update won column only where new data exists
        updated["won"] = updated["won_new"].combine_first(updated["won"])
        updated.drop(columns=["won_new"], inplace=True)

        # Compute profit
        updated["profit"] = updated.apply(self._compute_profit, axis=1)

        self.save(updated)
        logger.success("BetLogger: outcomes updated and profit recomputed.")

    # --------------------------------------------------------
    # Profit computation
    # --------------------------------------------------------
    @staticmethod
    def _compute_profit(row) -> Optional[float]:
        if pd.isna(row.get("won")):
            return None
        if row["won"] == 1:
            ml = row["ml"]
            stake = float(row["stake"])
            if ml > 0:
                dec = 1 + ml / 100.0
            else:
                dec = 1 + 100.0 / abs(ml)
            return stake * (dec - 1)
        else:
            return -float(row["stake"])

    # --------------------------------------------------------
    # Daily summary
    # --------------------------------------------------------
    def daily_summary(self, target_date: str) -> Dict[str, Any]:
        """
        Summarize all bets for a given prediction_date.
        """
        df = self.load()
        if df.empty:
            return {"bets": 0, "profit": 0.0, "roi": None}

        day_df = df[df["prediction_date"] == target_date].copy()
        if day_df.empty:
            return {"bets": 0, "profit": 0.0, "roi": None}

        # Only compute profit for resolved bets
        resolved = day_df.dropna(subset=["profit"])

        total_staked = resolved["stake"].sum()
        total_profit = resolved["profit"].sum()

        roi = total_profit / total_staked if total_staked > 0 else None

        summary = {
            "bets": len(day_df),
            "resolved": len(resolved),
            "profit": float(total_profit),
            "roi": roi,
        }

        logger.info(f"BetLogger: daily summary for {target_date}: {summary}")
        return summary


if __name__ == "__main__":
    logger = BetLogger()
    print(logger.daily_summary("2025-01-01"))
